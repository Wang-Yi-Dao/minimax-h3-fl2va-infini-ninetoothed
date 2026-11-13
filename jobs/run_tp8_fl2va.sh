#!/bin/bash
# nn38: nn30 + optional HIP_LAUNCH_BLOCKING switch (BLK=1). usage: nn38.sh [steps] [budget]
set +u
STEPS=${1:-${STEPS:-2}}
BUDGET=${2:-${BUDGET:-1650}}
T_START=$(date +%s)
echo "=== nn38 start $(date -u +%FT%T) steps=$STEPS budget=$BUDGET"
AFS=${IFAB_AFS_HOME:-$HOME}; H=$AFS/minimax_h3; B=${B:-$H/b1}
SH=/public/home/aclsylqidf/shared/MiniMax-H3/diffusers
PROJ=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$PROJ/.." && pwd)   # 仓库根（input/ 在这里）
W=${IFAB_WORKSPACE_ROOT:-$PWD}
[ -w "$W" ] || W=/dev/shm/mmh3
mkdir -p "$W/logs" "$W/out" "$H/logs" "$H/output2"
cd "$W" || exit 12
TS=$(date -u +%m%d_%H%M%S)
echo "GPU_COUNT=$(python3 -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null)"
echo "--- setup (cap 900s)"
timeout 900 python3 -u "$PROJ/setup.py" "$W" "$AFS" "$SH" > "$W/logs/nn38_setup.log" 2>&1
SE=$?
echo "SETUP_EXIT=$SE setup_secs=$(( $(date +%s)-T_START ))"
grep -E "^\[|STEP_FAIL|SETUP_OK|Traceback" "$W/logs/nn28_setup.log" | tail -10 | cut -c1-190
[ "$SE" = "0" ] || { echo "SETUP_FAILED"; tail -10 "$W/logs/nn28_setup.log" | cut -c1-190; echo "=== nn38 done $(date -u +%FT%T)"; exit 0; }
R="$W/InfiniCore/build/linux/x86_64/release"
L="$W/InfiniCore/python/infinicore/lib"
mkdir -p "$L"
# CRB_LIBS points at a from-scratch build of the same five libraries; when it is
# set the run uses those instead of the shipped ones (this is the rebuild check).
if [ -n "${CRB_LIBS:-}" ] && [ -d "${CRB_LIBS:-}" ]; then
  for f in _infinicore*.so libinfiniop.so libinfinicore_cpp_api.so libinfinirt.so libinfiniccl.so; do
    cp -f "$CRB_LIBS"/$f "$L"/ || echo FRESH_COPY_FAIL $f
  done
  echo "FRESH_LIBS_FROM=$CRB_LIBS"
  for f in libinfiniop.so libinfinicore_cpp_api.so libinfinirt.so libinfiniccl.so; do
    echo "FRESH $f $(stat -c%s "$L/$f" 2>/dev/null || echo 0)"
  done
else
cp -f "$R"/_infinicore*.so "$R"/libinfiniop.so "$R"/libinfinicore_cpp_api.so "$R"/libinfinirt.so "$R"/libinfiniccl.so "$L"/ 2>/dev/null
fi
if [ -z "${CRB_LIBS:-}" ] && [ -f "$B/out_release/libinfinicore_cpp_api.so" ]; then
  cp -f "$B/out_release/libinfinicore_cpp_api.so" "$L/libinfinicore_cpp_api.so"
  echo "PAD80_IN_LIB=$(nm -C "$L/libinfinicore_cpp_api.so" 2>/dev/null | grep -c h3_pad80)"
  echo "CAUSAL_IN_LIB=$(nm -C "$L/libinfinicore_cpp_api.so" 2>/dev/null | grep -c h3_causal)"
fi
export PYTHONPATH="$W/InfiniCore/python:$W/InfiniLM/python:$W/pylibs"
export LD_LIBRARY_PATH="$L:$W/.infini/lib:$W/fa:$R:/opt/dtk/lib:/opt/dtk/llvm/lib:/opt/hyhal/lib:/opt/dtk/.hyhal/rocm_smi:/opt/dtk/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}"
echo "--- hipCtxGetId shim (device-scoped kernel cache key)"
PRE=$B/pre_ctxid.so
if [ ! -f "$PRE" ]; then
  CCX=/opt/dtk/llvm/bin/clang; [ -x "$CCX" ] || CCX=/opt/dtk/bin/hipcc
  cat > "$W/pre_ctxid.c" <<'CEOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
typedef int hipError_t;
typedef void *hipCtx_t;
typedef hipError_t (*fn_getid)(hipCtx_t, unsigned long long *);
typedef hipError_t (*fn_getdev)(int *);
static fn_getid real_getid = 0;
static fn_getdev real_getdev = 0;
static void init_fns(void) {
    if (!real_getid) real_getid = (fn_getid)dlsym(RTLD_NEXT, "hipCtxGetId");
    if (!real_getdev) real_getdev = (fn_getdev)dlsym(RTLD_NEXT, "hipGetDevice");
}
hipError_t hipCtxGetId(hipCtx_t ctx, unsigned long long *id) {
    init_fns();
    hipError_t rc = real_getid ? real_getid(ctx, id) : -1;
    int dev = -1;
    if (real_getdev) (void)real_getdev(&dev);
    if (id) { unsigned long long base = *id; *id = base * 131ull + (unsigned long long)(dev + 1); }
    return rc;
}
CEOF
  timeout 180 "$CCX" -shared -fPIC -O2 -o "$PRE" "$W/pre_ctxid.c" -ldl > "$W/logs/pre_cc.log" 2>&1
  echo "PRE_CC_RC=$? bytes=$(stat -c%s "$PRE" 2>/dev/null || echo 0)"
fi
if [ -f "$PRE" ]; then
  export LD_PRELOAD="$PRE:${LD_PRELOAD:-}"
  echo "PRELOAD=$LD_PRELOAD"
fi
BLK=${BLK:-1}
if [ "$BLK" = "1" ]; then export HIP_LAUNCH_BLOCKING=1; echo "LAUNCH_BLOCKING=1"; else echo "LAUNCH_BLOCKING=0"; fi
export HF_HOME="$W/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FA_SO_PATH="$(cat "$W/fa_so_path.txt" 2>/dev/null)"
export PYTHONFAULTHANDLER=1
TP_IDS=${TP_IDS:-0,1,2,3,4,5,6,7}
cd "$W/InfiniLM" || exit 13
LOG="$W/logs/nn38_run.log"
rm -f "$LOG"
LEFT=$(( BUDGET - ( $(date +%s) - T_START ) ))
echo "RUN_BUDGET=${LEFT}s"
T0=$(date +%s)
# 首帧：优先仓库内 input/1280X1280.JPEG，其次平台个人存储 $HOME/input/
FFRAME="$REPO/input/1280X1280.JPEG"
[ -f "$FFRAME" ] || FFRAME="$H/input/1280X1280.JPEG"
[ -f "$FFRAME" ] || { echo "FIRST_FRAME_MISSING $FFRAME"; echo "=== nn38 done $(date -u +%FT%T)"; exit 0; }
echo "FIRST_FRAME=$FFRAME"
timeout $LEFT python3 -u examples/minimax_h3.py fl2va \
  --model-path "$W/model" --text "A dog falls asleep" \
  --first-frame "$FFRAME" --gpus "$TP_IDS" \
  --num-frames 124 --steps $STEPS --seed 42 \
  --output "$W/out/nl_$STEPS.mp4" > "$LOG" 2>&1 &
PY=$!
FAILED=0
LASTP=0
while kill -0 $PY 2>/dev/null; do
  sleep 5
  NOW=$(date +%s)
  if [ $((NOW-LASTP)) -ge 60 ]; then
    LASTP=$NOW
    echo "--- $(date -u +%T) progress ($((NOW-T0))s):"
    tr '\r' '\n' < "$LOG" 2>/dev/null | grep -v Materializing | grep -E "it/s|s/it|%\||Loading|Step " | tail -2 | cut -c1-140
  fi
  if grep -q -m1 -E "RankWorker stopped during run|Not Implemented|out of memory|Error in block" "$LOG" 2>/dev/null; then
    echo "FAIL_MARKER detected at $(date -u +%T)"
    tr '\r' '\n' < "$LOG" | grep -v Materializing | grep -n -i -E "error|not implemented|traceback|rankworker|exception during forward" | cut -c1-300 | head -10
    cp -f "$LOG" "$H/logs/nn38_${TS}_fail.log" 2>/dev/null
    kill -9 $PY 2>/dev/null
    FAILED=1
    break
  fi
done
wait $PY 2>/dev/null
RC=$?
RUN_S=$(( $(date +%s)-T0 ))
echo "RUN_RC=$RC failed=$FAILED run_secs=$RUN_S steps=$STEPS sec_per_step=$(python3 -c "print(round($RUN_S/max(1,$STEPS),2))")"
tr '\r' '\n' < "$LOG" | grep -v Materializing | grep -E "Error|error|Traceback|it/s|s/it" | tail -12 | cut -c1-200
cp -f "$LOG" "$H/logs/nn28_${TS}_run.log" 2>/dev/null
OUT="$W/out/nl_$STEPS.mp4"
if [ -f "$OUT" ]; then
  cp -f "$OUT" "$H/output2/nl${STEPS}_${TS}.mp4"
  echo "GET_MP4 bytes=$(stat -c%s "$OUT") path=$H/output2/nl${STEPS}_${TS}.mp4"
else
  echo "NO_MP4"
  echo "=== nn28 done $(date -u +%FT%T)"
  exit 0
fi
export PYTHONPATH="$H/pylibs_extra:${PYTHONPATH:-}"
if ! python3 -c "import skimage.metrics" 2>/dev/null; then
  echo "--- skimage missing, installing deps into pylibs_extra"
  mkdir -p "$W/shimp" "$H/pylibs_extra"
  cat > "$W/shimp/sitecustomize.py" <<'PY'
import builtins, io
_open = builtins.open
def _open_shim(file, *a, **k):
    try:
        return _open(file, *a, **k)
    except PermissionError:
        if str(file).endswith('/debian_version') or str(file).endswith('/lsb-release') or str(file).endswith('/os-release'):
            return io.StringIO("")
        raise
builtins.open = _open_shim
PY
  export PYTHONPATH="$W/shimp:$H/pylibs_extra:${PYTHONPATH:-}"
  timeout 400 python3 -m pip install --quiet --no-input --no-deps --target "$H/pylibs_extra" -i https://pypi.tuna.tsinghua.edu.cn/simple lazy_loader tifffile imageio scipy scikit-image 2>&1 | tail -3
  python3 -c "import skimage.metrics; print('SKIMAGE_OK')" 2>&1 | tail -2
fi
REF=""
for f in "$REPO/input/reference_fl2va.mp4" "$H/cmp/reference_fl2va.mp4" "$H/artifacts/reference_fl2va.mp4" "$H/input/reference_fl2va.mp4"; do
  [ -f "$f" ] && { REF="$f"; break; }
done
echo "REF=$REF"
LEFT2=$(( BUDGET + 90 - ( $(date +%s) - T_START ) ))
if [ -n "$REF" ] && [ "$LEFT2" -gt 60 ]; then
  python3 -c "import skimage, av" 2>/dev/null || echo "--- compare env still missing skimage/av"
  mkdir -p "$W/cmpout"
  timeout $LEFT2 python3 -u "$H/InfiniLM/examples/minimax_h3_compare.py" \
    --tasks fl2va --output-dir "$W/cmpout" --model-path "$W" \
    --reference-fl2va-video "$REF" --infinilm-fl2va-video "$OUT" 2>&1 | tee "$W/logs/nn38_cmp.log" | tail -25
  echo "CMP_RC=${PIPESTATUS[0]}"
  grep -E "Size:|Length:|Average SSIM:" "$W/logs/nn28_cmp.log" | cut -c1-190
  cp -f "$W/logs/nn28_cmp.log" "$H/logs/nn28_${TS}_cmp.log" 2>/dev/null
else
  echo "SKIP_CMP left=${LEFT2}s ref=${REF:-none}"
fi
echo "TOTAL_SECS=$(( $(date +%s)-T_START ))"
echo "=== nn28 done $(date -u +%FT%T)"