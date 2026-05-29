# Validate FID-faithful risk: g(sigma)=feature-space terminal sensitivity,
# d_s^phi = d_s*g, disc_FID = sum_gaps(incl terminal) mean(d_s^phi)*h^(p+1).
# Check ranking vs known FID on the panel.
import numpy as np, torch, torch.nn.functional as F
from cleanfid.features import build_feature_extractor
from autonomous_diffusion.samplers._seq_calib import build_reference
from autonomous_diffusion.samplers._common import denoise, karras_sigmas, resolve_shape
from autonomous_diffusion.samplers.proposed_control import load_calibration, calibration_cache_path, optimal_step_sigmas
from autonomous_diffusion.models import load_edm_network
from pathlib import Path
dev='cuda'; net=load_edm_network('cifar10',device=dev)
fe=build_feature_extractor('clean',device=dev,use_dataparallel=False)
calib=load_calibration(calibration_cache_path(net,root=Path('outputs/calibration'),calib_id='heun'))
sg=calib.sigma_grid; ds=calib.d_per_interval; p=2; mid=0.5*(sg[1:]+sg[:-1])
ref=build_reference(net,device=dev,B=32); sig_t=torch.tensor(ref.sig,dtype=torch.float64,device=dev)
smin,smax=float(ref.sig.min()),float(ref.sig.max())
@torch.no_grad()
def feats(x01):  # x in [-1,1] NCHW 32 -> features
    x=((x01+1)*127.5).clamp(0,255); x=F.interpolate(x.float(),size=299,mode='bicubic',align_corners=False)
    return fe(x)
@torch.no_grad()
def heun_to_zero(x, sigma_start, nsub=24):
    sigs=torch.exp(torch.linspace(np.log(float(sigma_start)),np.log(smin),nsub,device=dev,dtype=torch.float64))
    sigs=torch.cat([sigs,sigs.new_zeros(1)])
    for i in range(len(sigs)-1):
        sa,sb=sigs[i],sigs[i+1]; dn=denoise(net,x,sa); d=(x-dn)/sa
        if sb.item()==0: x=dn; break
        xe=x+(sb-sa)*d; dn2=denoise(net,xe,sb); d2=(xe-dn2)/sb; x=x+(sb-sa)*0.5*(d+d2)
    return x
# g(sigma_j) on a subset of dense nodes
@torch.no_grad()
def g_profile(node_idxs):
    gen=torch.Generator(device=dev).manual_seed(7); gs={}
    for j in node_idxs:
        xj=ref.Xref[j]; sj=sig_t[j]
        x0_ref=heun_to_zero(xj,sj); f_ref=feats(x0_ref)
        eps=1e-2*float(sj)
        v=torch.randn(xj.shape,generator=gen,device=dev,dtype=torch.float64); v=v/v.reshape(v.shape[0],-1).norm(dim=1).view(-1,1,1,1)
        x0_p=heun_to_zero(xj+eps*v,sj); f_p=feats(x0_p)
        gs[j]=((f_p-f_ref).pow(2).sum(1).mean().item())/(eps**2)
    return gs
M=len(ref.sig); idxs=list(range(0,M,max(M//28,1)))
gp=g_profile(idxs)
gi=np.array(sorted(gp.keys())); gv=np.array([gp[i] for i in gi])
gsig=ref.sig[gi]
def g_at(s): return float(np.interp(np.log(max(s,smin)), np.log(gsig[::-1]), gv[::-1]))
def dmean(lo,hi):
    m=(mid>=lo)&(mid<=hi); return float(ds[m].mean()) if m.any() else float(ds[np.argmin(np.abs(mid-0.5*(lo+hi)))])
def discFID(grid):
    s=[x for x in grid if x>0]; tot=0.0
    gaps=[(s[i],s[i+1]) for i in range(len(s)-1)]+[(s[-1],0.0)]
    for a,b in gaps:
        lo,hi=min(a,b),max(a,b); h=hi-lo; smid=np.sqrt(max(a*b,smin*smin)) if b>0 else max(0.5*a,smin)
        tot+=g_at(smid)*dmean(max(lo,smin),hi)*h**(p+1)
    return tot
panel={'good_pointwise':optimal_step_sigmas(calib,5,p=2,perceptual_weight_k=2.0).tolist(),
 'karras_rho7':karras_sigmas(5,smin,smax,device='cpu').cpu().numpy().tolist(),
 'cluster_hi':[80.0,20.8,20.85,20.87,20.88,0.0],'refined_lowQ':[80.0,29.24,10.82,3.45,0.79,0.0],
 'forced_smin':[80.0,20.41,4.0,0.38,0.002,0.0]}
fids={'good_pointwise':13.16,'karras_rho7':44.90,'cluster_hi':270.30,'refined_lowQ':26.71,'forced_smin':87.85}
print('g(sigma): hi-sigma',round(g_at(40),3),' mid',round(g_at(1),3),' lo',round(g_at(0.05),3))
rows=[(n,discFID(g),fids[n]) for n,g in panel.items()]
print(f'{"grid":16s} {"disc_FID":>12s} {"FID":>8s}')
for n,d,f in rows: print(f'{n:16s} {d:12.4f} {f:8.2f}')
print('rank disc_FID:',[r[0] for r in sorted(rows,key=lambda r:r[1])])
print('rank FID     :',[r[0] for r in sorted(rows,key=lambda r:r[2])])
