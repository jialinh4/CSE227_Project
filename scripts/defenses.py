# scripts/defenses.py
import os, cv2, numpy as np
from pathlib import Path

def load_alpha(p):
    a = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if a is None: raise FileNotFoundError(p)
    if a.ndim==3: a = a[...,0]
    a = a.astype(np.float32)
    if a.max()>1.5: a/=255.0
    return a

def save_alpha(a, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    cv2.imwrite(p, (np.clip(a,0,1)*255).astype(np.uint8))

def erosion(a, radius=5):
    k=2*radius+1
    ker=np.ones((k,k),np.uint8)
    return cv2.erode((a*255).astype(np.uint8), ker).astype(np.float32)/255.0

def jitter(a, max_shift=2, prob=0.6):
    h,w=a.shape; out=a.copy()
    inner=(a>0.8).astype(np.uint8); outer=(a>0.05).astype(np.uint8)
    ring=(outer-inner).clip(0,1)
    ys,xs=np.where(ring>0)
    for y,x in zip(ys,xs):
        if np.random.rand()>prob: continue
        ny=np.clip(y+np.random.randint(-max_shift,max_shift+1),0,h-1)
        nx=np.clip(x+np.random.randint(-max_shift,max_shift+1),0,w-1)
        out[y,x]=a[ny,nx]
    return out

def bnoise(a, sigma=0.04):
    inner=(a>0.8).astype(np.uint8); outer=(a>0.02).astype(np.uint8)
    ring=(outer-inner).clip(0,1)
    return np.clip(a + ring*np.random.randn(*a.shape)*sigma, 0, 1)

def apply(mask_dir, out_dir, method, **kw):
    os.makedirs(out_dir, exist_ok=True)
    for p in sorted(Path(mask_dir).glob("*.png")):
        a=load_alpha(str(p))
        if method=="erosion": a2=erosion(a, kw.get("radius",5))
        elif method=="jitter": a2=jitter(a, kw.get("max_shift",2), kw.get("prob",0.6))
        elif method=="noise":  a2=bnoise(a, kw.get("sigma",0.04))
        else: raise ValueError("unknown method")
        save_alpha(a2, os.path.join(out_dir, p.name))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--method", choices=["erosion","jitter","noise"], required=True)
    ap.add_argument("--radius", type=int, default=5)
    ap.add_argument("--max_shift", type=int, default=2)
    ap.add_argument("--prob", type=float, default=0.6)
    ap.add_argument("--sigma", type=float, default=0.04)
    args = ap.parse_args()

    # 只把与该方法相关的参数传进去，避免重复传参
    kwargs = {}
    if args.method == "erosion":
        kwargs["radius"] = args.radius
    elif args.method == "jitter":
        kwargs["max_shift"] = args.max_shift
        kwargs["prob"] = args.prob
    elif args.method == "noise":
        kwargs["sigma"] = args.sigma

    apply(args.mask_dir, args.out_dir, args.method, **kwargs)
    print("✔ saved to", args.out_dir)

    #后期可选加入合成防御后的视频