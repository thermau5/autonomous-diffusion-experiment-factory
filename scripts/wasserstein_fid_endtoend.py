import numpy as np, torch, torch.nn.functional as F
from cleanfid.features import build_feature_extractor
from autonomous_diffusion.samplers._seq_calib import build_reference
from autonomous_diffusion.samplers._common import denoise, sample_initial_noise, resolve_shape
from autonomous_diffusion.samplers.proposed_control import load_calibration, calibration_cache_path
from autonomous_diffusion.models import load_edm_network
from autonomous_diffusion.metrics import CleanFIDConfig, compute_clean_fid
from pathlib import Path
dev='cuda'; net=load_edm_network('cifar10',device=dev); shape=resolve_shape(net,None)
cfg=CleanFIDConfig(dataset_name='cifar10',dataset_res=32,dataset_split='train',mode='clean')
fe=build_feature_extractor('clean',device=dev,use_dataparallel=False)
ref=build_reference(net,device=dev,B=64); sig_t=torch.tensor(ref.sig,dtype=torch.float64,device=dev)
smin,smax=float(ref.sig.min()),float(ref.sig.max())
cal=load_calibration(calibration_cache_path(net,root=Path('outputs/calibration'),calib_id='unipc')); p=2
sg=cal.sigma_grid; ds=cal.d_per_interval; mid=0.5*(sg[:-1]+sg[1:]); ints=np.abs(sg[:-1]-sg[1:])
@torch.no_grad()
def feats(x01):
    x=((x01+1)*127.5).clamp(0,255); x=F.interpolate(x.float(),size=299,mode='bicubic',align_corners=False); return fe(x)
@torch.no_grad()
def heun_to_zero(x,s0,nsub=32):
    sgr=torch.exp(torch.linspace(np.log(float(s0)),np.log(smin),nsub,device=dev,dtype=torch.float64)); sgr=torch.cat([sgr,sgr.new_zeros(1)])
    for i in range(len(sgr)-1):
        sa,sb=sgr[i],sgr[i+1]; dn=denoise(net,x,sa); d=(x-dn)/sa
        if sb.item()==0: x=dn; break
        xe=x+(sb-sa)*d; dn2=denoise(net,xe,sb); d2=(xe-dn2)/sb; x=x+(sb-sa)*0.5*(d+d2)
    return x
def near(s): return int(np.argmin(np.abs(ref.sig-s)))
# DIRECT feature one-step difficulty: real Heun step of size h at sigma, then true flow to 0, ||dphi||^2 / h^(p+1)
@torch.no_grad()
def dsphi_direct(nodes):
    out={}
    for j in nodes:
        sj=sig_t[j]
        # reference step size: 1.5x in sigma (a real finite step), but keep in-range
        sj2=max(float(sj)*0.6, smin*1.5)
        h=float(sj)-sj2
        if h<=0: continue
        xj=ref.Xref[j]
        x0_true=feats(heun_to_zero(xj,float(sj)))
        # one Heun step sj->sj2
        dn=denoise(net,xj,sj); d=(xj-dn)/sj; xe=xj+(sj2-sj)*d
        st=torch.full((xj.shape[0],),sj2,device=dev,dtype=torch.float64); dn2=(xe-denoise(net,xe,st))/sj2; xs=xj+(sj2-sj)*0.5*(d+dn2)
        x0_disc=feats(heun_to_zero(xs,sj2))
        out[j]=((x0_disc-x0_true).pow(2).sum(1).mean().item())/(h**(p+1))
    return out
M=len(ref.sig); nodes=list(range(0,M-2,max(M//36,1)))
dphi=dsphi_direct(nodes); ni=np.array(sorted(dphi)); nv=np.array([dphi[i] for i in ni]); nsig=ref.sig[ni]
import math
def dphi_at(s): return float(np.interp(np.log(np.clip(s,smin,smax)),np.log(nsig[::-1]),nv[::-1]))
print(f'dsphi_direct: 40->{dphi_at(40):.3g} 1->{dphi_at(1):.3g} 0.05->{dphi_at(0.05):.3g}')
def mstar(weight_at,K):
    w=np.array([weight_at(m) for m in mid]); wt=(np.clip(ds,1e-12,None)*w)**(1.0/(p+1))*ints
    cum=np.concatenate([[0.0],np.cumsum(wt)]); tgt=np.linspace(0,cum[-1],K+1); ns=np.interp(tgt,cum,sg); ns[0]=smax; ns[-1]=0.0; return ns.astype(np.float32)
@torch.no_grad()
def run_unipc(grid,N=10000,seed=0,bs=64):
    ss=torch.tensor(grid,dtype=torch.float32,device=dev); K=ss.shape[0]-1; out=[];done=0
    while done<N:
        b=min(bs,N-done); x=sample_initial_noise((b,*shape),float(ss[0]),seed=seed+done,device=dev); dc=denoise(net,x,ss[0]); dp=None
        for i in range(K):
            si,sn=ss[i],ss[i+1]
            if sn.item()==0: x=dc; break
            if dp is None: h=si.log()-sn.log(); xp=(sn/si)*x-torch.expm1(-h)*dc
            else:
                spv=ss[i-1]; h1=spv.log()-si.log(); h=si.log()-sn.log(); r=h1/h; Dp=(1+1/(2*r))*dc-(1/(2*r))*dp; xp=(sn/si)*x-torch.expm1(-h)*Dp
            dn=denoise(net,xp,sn); Dc=0.5*dc+0.5*dn; he=si.log()-sn.log(); x=(sn/si)*x-torch.expm1(-he)*Dc; dp=dc; dc=dn
        out.append(x.clamp(-1,1).cpu()); done+=b
    return torch.cat(out,0)[:N]
def fid3(g): return float(np.mean([compute_clean_fid(((run_unipc(g,seed=s)+1)*127.5).clamp(0,255).to(torch.uint8).numpy(),cfg)['fid'] for s in (0,1,2)]))
print('--- CHECK 1: builder w=sigma^-2 should reproduce locked proposed_unipc ---')
for K in [5,8]:
    g=mstar(lambda s:s**(-2.0),K); print(f'  K={K} builder-k2 FID={fid3(g):.3f}  (locked {21.46 if K==5 else 9.18})')
print('--- CHECK 2: m* from DIRECT feature one-step difficulty ---')
for K in [5,8]:
    g=mstar(dphi_at,K); print(f'  K={K} m*_dsphi FID={fid3(g):.3f}  grid[:5]={[round(x,4) for x in g.tolist()[:5]]}')
