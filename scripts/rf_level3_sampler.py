import sys, argparse, numpy as np, torch
RF='/home/thomaslin/Autonomous-Diffusion/third_party/rectified_flow/ImageGeneration'
sys.path.insert(0, RF)
sys.path.insert(0, '/home/thomaslin/Autonomous-Diffusion/src')
from configs.rectified_flow import cifar10_rf_gaussian_ddpmpp as cfgmod
from models import utils as mutils
import models.ncsnpp
from autonomous_diffusion.metrics import CleanFIDConfig, compute_clean_fid

def build_model():
    config = cfgmod.get_config()
    model = mutils.create_model(config)
    ck = torch.load(RF+'/logs/1_rectified_flow/checkpoints/checkpoint_8.pth', map_location='cuda', weights_only=False)
    inner = model.module if hasattr(model,'module') else model
    # use EMA shadow params (best for sampling)
    ema_keys = ck['ema']['shadow_params']
    msd = {k.replace('module.',''):v for k,v in ck['model'].items()}
    inner.load_state_dict(msd, strict=True)
    # apply EMA
    params = [p for p in inner.parameters() if p.requires_grad]
    if len(ema_keys)==len(params):
        for p,s in zip(params, ema_keys): p.data.copy_(s.data.to(p.device))
        ema_applied=True
    else:
        ema_applied=False
    inner.eval()
    return inner, ema_applied

@torch.no_grad()
def euler_rf(model, N, n_samples=10000, seed=0, bs=500, eps=1e-3):
    out=[]; done=0
    while done<n_samples:
        b=min(bs,n_samples-done)
        g=torch.Generator(device='cuda').manual_seed(seed*100000+done)
        x=torch.randn(b,3,32,32,generator=g,device='cuda')  # noise_scale=1.0
        dt=1.0/N
        for i in range(N):
            num_t=i/N*(1.0-eps)+eps
            t=torch.ones(b,device='cuda')*num_t
            v=model(x, t*999)
            x=x + v*dt
        out.append(x.clamp(-1,1).cpu()); done+=b
    return torch.cat(out,0)[:n_samples]

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--nfe',type=int,default=8); ap.add_argument('--n',type=int,default=64); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--fid',action='store_true')
    a=ap.parse_args()
    model,ema=build_model(); print('EMA applied:',ema)
    s=euler_rf(model,a.nfe,n_samples=a.n,seed=a.seed)
    print(f'NFE={a.nfe} samples={tuple(s.shape)} range=[{s.min():.3f},{s.max():.3f}] mean={s.mean():.3f}')
    if a.fid:
        cfg=CleanFIDConfig(dataset_name='cifar10',dataset_res=32,dataset_split='train',mode='clean')
        arr=((s+1)*127.5).clamp(0,255).to(torch.uint8).numpy()
        print(f'  Clean-FID(NFE={a.nfe}, n={a.n}) = {compute_clean_fid(arr,cfg)["fid"]:.3f}')
