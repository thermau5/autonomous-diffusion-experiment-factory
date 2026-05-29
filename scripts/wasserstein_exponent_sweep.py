import numpy as np, torch, os
from autonomous_diffusion.samplers._common import denoise, sample_initial_noise, resolve_shape
from autonomous_diffusion.samplers.proposed_control import load_calibration, calibration_cache_path
from autonomous_diffusion.models import load_edm_network
from autonomous_diffusion.metrics import CleanFIDConfig, compute_clean_fid
from pathlib import Path
dev='cuda'; net=load_edm_network('cifar10',device=dev); shape=resolve_shape(net,None)
cfg=CleanFIDConfig(dataset_name='cifar10',dataset_res=32,dataset_split='train',mode='clean')
cal=load_calibration(calibration_cache_path(net,root=Path('outputs/calibration'),calib_id='unipc'))
sg=cal.sigma_grid; ds=cal.d_per_interval; mid=0.5*(sg[:-1]+sg[1:]); ints=np.abs(sg[:-1]-sg[1:])
smin,smax=float(sg.min()),float(sg.max())
gp=np.load('/home/thomaslin/.claude/jobs/b3e9d3c7/g_profile.npz'); gsig=gp['gsig']; gv=gp['gv']
def g_at(s): return float(np.interp(np.log(np.clip(s,smin,smax)),np.log(gsig[::-1]),gv[::-1]))
def mstar(weight_fn,K,p):
    w=np.array([weight_fn(m) for m in mid]); wt=(np.clip(ds,1e-12,None)*w)**(1.0/(p+1))*ints
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
wg=g_at; w2=lambda s:s**(-2.0)
K=5
print(f'locked (sigma^-2, p=2) reference = 21.46')
for name,wf,p in [('g, p=2',wg,2),('g, p=4',wg,4),('g, p=3',wg,3),('sigma^-2, p=4',w2,4),('sigma^-2, p=2 (builder check)',w2,2)]:
    g=mstar(wf,K,p); print(f'  {name:30s} K=5 FID={fid3(g):.3f}  grid[:5]={[round(x,4) for x in g.tolist()[:5]]}')
