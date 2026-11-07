# kb7: fix the ninetoothed AOT kernel-cache key.  The generated launch functions do
#      "unsigned long long ctx_id; (void)hipCtxGetId(NULL, &ctx_id);" -- on this DTK
#      stack that value is identical for every device, so one cached Kernel (module
#      loaded in the first device's context) is reused on all devices and every later
#      device computes garbage.  We key the map by the current device instead.
#      usage: kb7.py <B> patch|check [NT_DIR]
import os, sys, glob

B = sys.argv[1].rstrip("/")
MODE = sys.argv[2] if len(sys.argv) > 2 else "patch"
D = sys.argv[3] if len(sys.argv) > 3 else os.path.join(B, "ic/build/ninetoothed")

DECL = "    unsigned long long ctx_id;"
GET = "    (void)hipCtxGetId(NULL, &ctx_id);"
NEW = ("    int h3_dev_ = -1;\n"
       "    (void)hipGetDevice(&h3_dev_);\n"
       "    if (h3_dev_ < 0) { h3_dev_ = 0; }\n"
       "    unsigned long long ctx_id = (unsigned long long)(h3_dev_ + 1);")

def log(m):
    print(m, flush=True)

files = sorted(glob.glob(os.path.join(D, "*.cpp")))
log("NT_DIR=%s CPP_FILES=%d" % (D, len(files)))
hav = [p for p in files if GET in open(p, encoding="utf-8", errors="replace").read()]
log("CPP_WITH_HIPCTXGETID=%d" % len(hav))
done_all = [p for p in files
            if "h3_dev_" in open(p, encoding="utf-8", errors="replace").read()]
log("CPP_WITH_DEVKEY=%d" % len(done_all))
if not hav and len(done_all) >= 100:
    log("ALREADY_ALL=%d" % len(done_all))
    sys.exit(0)

HIPV = ("    int h3_dev_ = -1;\n"
        "    (void)hipGetDevice(&h3_dev_);\n"
        "    if (h3_dev_ < 0) { h3_dev_ = 0; }\n"
        "    unsigned long long ctx_id = (unsigned long long)(h3_dev_ + 1);")
CUV = ("    CUdevice h3_dev_ = 0;\n"
       "    (void)cuCtxGetDevice(&h3_dev_);\n"
       "    unsigned long long ctx_id = (unsigned long long)(h3_dev_ + 1);")

if MODE == "cufix":
    n = 0
    for p in files:
        s = open(p, encoding="utf-8", errors="replace").read()
        if HIPV in s:
            open(p, "w", encoding="utf-8").write(s.replace(HIPV, CUV, 1))
            n += 1
    log("CUFIX=%d" % n)
    sys.exit(0)

if MODE == "verify":
    p0 = files[0]
    ls = open(p0, encoding="utf-8", errors="replace").read().split("\n")
    for i, ln in enumerate(ls):
        if "h3_dev_" in ln:
            for j in range(max(0, i - 3), min(len(ls), i + 6)):
                log("V%d: %s" % (j + 1, ls[j][:220]))
            break
    sys.exit(0)

if MODE == "check":
    done = sum(1 for p in hav if "h3_dev_" in open(p, encoding="utf-8", errors="replace").read())
    log("PATCHED_NOW=%d OF %d" % (done, len(hav)))
    sys.exit(0 if done == len(hav) and hav else 7)

bn = os.path.join(B, "nt_backup3")
os.makedirs(bn, exist_ok=True)
patched = already = miss = 0
for p in hav:
    s = open(p, encoding="utf-8", errors="replace").read()
    if "h3_dev_" in s:
        already += 1
        continue
    dst = os.path.join(bn, os.path.basename(p))
    if not os.path.exists(dst):
        open(dst, "w", encoding="utf-8").write(s)
    if (DECL + "\n" + GET) in s:
        s2 = s.replace(DECL + "\n" + GET, NEW, 1)
    else:
        s2 = s.replace(GET, "", 1).replace(DECL, NEW, 1)
    if s2 == s or "h3_dev_" not in s2:
        miss += 1
        log("MISS %s" % os.path.basename(p))
        continue
    open(p, "w", encoding="utf-8").write(s2)
    patched += 1

log("PATCHED=%d ALREADY=%d MISS=%d BACKUP=%s" % (patched, already, miss, bn))
if hav:
    p0 = hav[0]
    for i, ln in enumerate(open(p0, encoding="utf-8", errors="replace").read().split("\n")):
        if "h3_dev_" in ln or "ctx_id]" in ln:
            log("SAMPLE L%d: %s" % (i + 1, ln[:200]))
sys.exit(0 if patched + already > 0 and miss == 0 else 8)
