# Directly measure the discretization Wasserstein risk R_disc =
# feature-space Frechet distance between K-step samples and infinity-step
# (converged) samples. Compare to FID(vs data) and to our bound Q.
import numpy as np, torch, torch.nn.functional as F
from scipy import linalg
from cleanfid.features import build_feature_extractor
from autonomous_diffusion.samplers._seq_calib import build_reference
from autonomous_diffusion.samplers._common import denoise, sample_initial_noise, resolve_shape
from autonomous_diffusion.samplers.proposed_control import load_calibration, calibration_cache_path, optimal_step_sigmas
from autonomous_diffusion.samplers._common import karras_sigmas
from autonomous_diffusion.models import load_edm_network
from pathlib import Path
dev='cuda'; net=load_edm_network('cifar10',device=dev); shape=resolve_shape(net,None)
fe=build_feature_extractor('clean',device=dev,use_dataparallel=False)
smin,smax=0.002,80.0
@torch.no_grad()
def feats_of(x01):
    x=((x01+1)*127.5).clamp(0,255); x=F.interpolate(x.float(),size=299,mode='bicubic',align_corners=False); return fe(x).cpu().numpy()
@torch.no_grad()
def heun(grid,N=10000,seed=0,bs=64):
    ss=torch.tensor(grid,dtype=torch.float32,device=dev); n=ss.shape[0]-1; out=[];done=0
    while done<N:
        b=min(bs,N-done); x=sample_initial_noise((b,*shape),float(ss[0]),seed=seed+done,device=dev)
        for i in range(n):
            si,sn=ss[i],ss[i+1]; dn=denoise(net,x,si); d=(x-dn)/si
            if sn.item()==0: x=dn; break
            xe=x+(sn-si)*d; dn2=denoise(net,xe,sn); d2=(xe-dn2)/sn; x=x+(sn-si)*0.5*(d+d2)
        out.append(x.clamp(-1,1).cpu()); done+=b
    return torch.cat(out,0)[:N]
def feats_batched(samp,bs=256):
    fs=[]
    for i in range(0,samp.shape[0],bs): fs.append(feats_of(samp[i:i+bs].to(dev)))
    return np.concatenate(fs,0)
def frechet(f1,f2):
    m1,m2=f1.mean(0),f2.mean(0); s1=np.cov(f1,rowvar=False); s2=np.cov(f2,rowvar=False)
    cov,_=linalg.sqrtm(s1@s2,disp=False); cov=cov.real
    return float(((m1-m2)**2).sum()+np.trace(s1+s2-2*cov))
# infinity-step reference: Heun 128 steps (substantive)+0
refgrid=np.concatenate([karras_sigmas(128,smin,smax,device='cpu').cpu().numpy()[:-1],[0.0]])
print('generating inf-step reference (Heun-128, 10k)...'); ref_s=heun(refgrid.tolist()); ref_f=feats_batched(ref_s)
calib=load_calibration(calibration_cache_path(net,root=Path('outputs/calibration'),calib_id='heun'))
panel={'good':optimal_step_sigmas(calib,5,p=2,perceptual_weight_k=2.0).tolist(),
 'karras':karras_sigmas(5,smin,smax,device='cpu').cpu().numpy().tolist(),
 'cluster':[80.0,20.8,20.85,20.87,20.88,0.0],'refined':[80.0,29.24,10.82,3.45,0.79,0.0],
 'forced':[80.0,20.41,4.0,0.38,0.002,0.0]}
fid_data={'good':13.16,'karras':44.90,'cluster':270.30,'refined':26.71,'forced':87.85}
Qpix={'good':26.03,'karras':64.11,'cluster':55.89,'refined':12.97,'forced':26.41}  # pixel bound, terminal-incl
print(f'{"grid":10s} {"R_disc(feat-W2 vs inf)":>22s} {"FID(vs data)":>13s} {"Q_pixel_bound":>14s}')
rows=[]
for n,g in panel.items():
    s=heun(list(g)); R=frechet(feats_batched(s),ref_f); rows.append((n,R,fid_data[n],Qpix[n]))
    print(f'{n:10s} {R:22.3f} {fid_data[n]:13.2f} {Qpix[n]:14.3f}')
print('rank R_disc :',[r[0] for r in sorted(rows,key=lambda r:r[1])])
print('rank FID    :',[r[0] for r in sorted(rows,key=lambda r:r[2])])
print('rank Q_pixel:',[r[0] for r in sorted(rows,key=lambda r:r[3])])
