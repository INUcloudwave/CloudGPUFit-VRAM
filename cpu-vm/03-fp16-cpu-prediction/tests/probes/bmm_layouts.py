import torch, time
torch.set_num_threads(8)
def bench(name, f, flops, n=3):
    f(); t=time.time()
    for _ in range(n): f()
    s=(time.time()-t)/n; print(f"{name}: {flops/s/1e9:.2f} GFLOPS")
B,M,K,N=64,128,64,128
for dt in (torch.float16,):
    a=torch.randn(B,M,K,dtype=dt); b=torch.randn(B,K,N,dtype=dt)
    bt=torch.randn(B,N,K,dtype=dt).transpose(1,2); at=torch.randn(B,K,M,dtype=dt).transpose(1,2)
    fl=2*B*M*K*N
    bench("bmm a@b contig", lambda: torch.bmm(a,b), fl)
    bench("bmm a@bT(view)", lambda: torch.bmm(a,bt), fl)
    bench("bmm aT@b", lambda: torch.bmm(at,b), fl)
    bench("bmm aT@bT", lambda: torch.bmm(at,bt), fl)
    import numpy as np
    def npb():
        out=torch.empty(B,M,N,dtype=dt)
        out.numpy()[:]=np.matmul(a.numpy().astype(np.float32), b.numpy().astype(np.float32))
        return out
    bench("numpy fp32 bmm", npb, fl)
    a1=torch.randn(B,1,K,dtype=dt); b1=torch.randn(B,K,1024,dtype=dt)
    bench("decode-like bmm (B,1,K)@(B,K,1024)", lambda: torch.bmm(a1,b1), 2*B*K*1024)
    bench("decode-like numpy", lambda: np.matmul(a1.numpy().astype(np.float32), b1.numpy().astype(np.float32)), 2*B*K*1024)
    w=torch.randn(512,2048,dtype=dt); x=torch.randn(64,512,dtype=dt)
    bench("mm x@w (row-major w)", lambda: torch.mm(x,w), 2*64*512*2048)
    bench("mm x@wT(view)", lambda: torch.mm(x,torch.randn(2048,512,dtype=dt).t()), 2*64*512*2048)
