
import os, sys, math
import torch, infinicore
import torch.nn.functional as F

BF = infinicore.bfloat16
D = 128
SCALE = D ** -0.5

def log(*a):
    print(*a, flush=True)

def use(i):
    torch.cuda.set_device(i)
    infinicore.set_device(infinicore.device("cuda", i))
    infinicore.sync_stream()
    return infinicore.device("cuda", i)

def make(i, H, S, seed=101):
    g = torch.Generator(device="cpu").manual_seed(seed)
    hd = H * D
    fq = (torch.randn(1, S, 3 * hd, generator=g) * 0.6).to(torch.float16).to(torch.bfloat16).cuda(i)
    return [p.view(1, S, H, D) for p in (fq[:, :, :hd], fq[:, :, hd:2*hd], fq[:, :, 2*hd:])]

def ic_t(x, dv):
    return infinicore.strided_from_blob(x.data_ptr(), list(x.shape), list(x.stride()), dtype=BF, device=dv)

def run(i, H, S):
    dv = use(i)
    q, k, v = make(i, H, S)
    o = infinicore.mha(ic_t(q, dv), ic_t(k, dv), ic_t(v, dv), None, SCALE, False)
    infinicore.sync_stream()
    ob = torch.zeros(tuple(o.shape), dtype=torch.bfloat16, device="cuda:%d" % i)
    infinicore.from_blob(ob.data_ptr(), list(ob.shape), dtype=BF, device=dv).copy_(o)
    infinicore.sync_stream()
    qh, kh, vh = q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3)
    r = F.scaled_dot_product_attention(qh, kh, vh, scale=SCALE).permute(0,2,1,3)
    a, b = ob.float(), r.float()
    d = (a - b).abs()
    cs = float((a*b).sum() / (a.norm()*b.norm() + 1e-12))
    log("FA_S S=%6d H=%2d dev=%d max=%.6g mean=%.6g cos=%.6f std=%.6g std_ref=%.6g relmean=%.3e" % (
        S, H, i, d.max().item(), d.mean().item(), cs, a.std().item(), b.std().item(),
        (d.mean().item() / (b.abs().mean().item() + 1e-9))))
    del q, k, v, ob, r, a, b, d, qh, kh, vh
    torch.cuda.empty_cache()

def main():
    log("NN36FA devices=%d" % torch.cuda.device_count())
    for (H, S) in [(7, 4096), (7, 27570), (28, 27570)]:
        try:
            run(0, H, S)
        except Exception as e:
            log("FA_S S=%6d H=%2d dev=0 FAILED %s: %s" % (S, H, type(e).__name__, str(e)[:160]))
    log("NN36FA_DONE")

main()
