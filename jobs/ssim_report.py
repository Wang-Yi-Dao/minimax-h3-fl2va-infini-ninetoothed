
import os, sys, glob, numpy as np
from skimage.metrics import structural_similarity as sk
from PIL import Image
import av
H = os.path.join(os.environ.get("IFAB_AFS_HOME") or os.path.expanduser("~"), "minimax_h3")
W = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
def prep(a, m=512):
    im = Image.fromarray(a).convert("L")
    s = min(1.0, m/max(im.size))
    if s < 1.0:
        im = im.resize((max(1,round(im.width*s)), max(1,round(im.height*s))), Image.Resampling.BILINEAR)
    return np.asarray(im)
def load(p, step=6):
    d = {}
    with av.open(p) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i % step == 0: d[i] = f.to_ndarray(format="rgb24")
    return d
REF_REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input", "reference_fl2va.mp4")
files = {"ref50": REF_REPO if os.path.isfile(REF_REPO) else os.path.join(H, "cmp/reference_fl2va.mp4")}
for p in sorted(glob.glob(os.path.join(W, "out", "*.mp4"))):
    files[os.path.basename(p)[:-4]] = p
newest = {}
for p in sorted(glob.glob(os.path.join(H, "output2", "*.mp4"))):
    base = os.path.basename(p)[:-4]
    tag = base.rsplit("_", 2)[0]
    newest[tag] = p
for k, v in newest.items():
    files.setdefault("hist_" + k, v)
D = {}
for k, v in files.items():
    if os.path.exists(v):
        D[k] = load(v)
print("VIDS", {k: len(v) for k, v in D.items()})
for k, v in D.items():
    ks = sorted(v)
    if ks:
        f0 = v[ks[0]].astype(np.float32)
        print("STAT %-16s frames=%d f0_mean=%.2f f0_std=%.2f" % (k, len(ks), f0.mean(), f0.std()))
keys = sorted(D)
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        a, b = keys[i], keys[j]
        idx = sorted(set(D[a]) & set(D[b]))
        if not idx: print("NODATA", a, b); continue
        s = [float(sk(prep(D[a][k]), prep(D[b][k]), data_range=255)) for k in idx]
        mad = [float(np.abs(D[a][k].astype(np.float32)-D[b][k].astype(np.float32)).mean()) for k in idx]
        print("SSIM %-20s vs %-20s n=%d mean=%.6f min=%.4f mad=%.2f" % (a, b, len(s), np.mean(s), min(s), np.mean(mad)))
for pair in os.environ.get("PERSERIES", "").split(","):
    if "|" not in pair: continue
    a, b = pair.split("|")
    if a not in D or b not in D: print("PERSERIES_MISSING", pair); continue
    idx = sorted(set(D[a]) & set(D[b]))
    s = [float(sk(prep(D[a][k]), prep(D[b][k]), data_range=255)) for k in idx]
    print("PSER %s|%s" % (a, b))
    print("  frames " + " ".join(str(k) for k in idx))
    print("  ssim   " + " ".join("%.3f" % x for x in s))
print("PROBE3_DONE")
