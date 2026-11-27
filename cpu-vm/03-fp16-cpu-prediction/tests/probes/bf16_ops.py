import torch, time
import torch.nn.functional as F
torch.set_num_threads(8)
def bench(name, f, flops, n=3):
    f(); t=time.time()
    for _ in range(n): f()
    s=(time.time()-t)/n; print(f"{name}: {flops/s/1e9:.2f} GFLOPS")
for dt in (torch.float32, torch.bfloat16, torch.float16):
    a=torch.randn(64,128,64,dtype=dt); b=torch.randn(64,64,128,dtype=dt)
    bench(f"{dt} bmm", lambda: torch.bmm(a,b), 2*64*128*64*128)
    a1=torch.randn(512,1,64,dtype=dt); b1=torch.randn(512,64,1024,dtype=dt).transpose(1,2).contiguous().transpose(1,2)
    bench(f"{dt} decode bmm", lambda: torch.bmm(a1,b1), 2*512*64*1024)
    x=torch.randn(512,1024,dtype=dt); w=torch.randn(1024,1024,dtype=dt)
    bench(f"{dt} mm rowmajor", lambda: torch.mm(x,w), 2*512*1024*1024)
    bench(f"{dt} linear", lambda: F.linear(x,w), 2*512*1024*1024)
    img=torch.randn(8,64,56,56,dtype=dt); cw=torch.randn(64,64,3,3,dtype=dt)
    bench(f"{dt} conv2d", lambda: F.conv2d(img,cw,padding=1), 2*8*64*56*56*64*9, n=1)
