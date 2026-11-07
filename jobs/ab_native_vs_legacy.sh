#!/bin/bash
# nn34: 1-step A/B. TP8 (InfiniLM, LD_PRELOAD shim) 1 step vs legacy PyTorch 1 step, same seed/prompt.
set +u
STEPS=${1:-${STEPS:-1}}
BUDGET=${2:-${BUDGET:-1500}}
T_START=$(date +%s)
echo "=== nn34 start $(date -u +%FT%T) steps=$STEPS budget=$BUDGET"
AFS=${IFAB_AFS_HOME:-$HOME}; H=$AFS/minimax_h3; B=$H/b1
SH=/public/home/aclsylqidf/shared/MiniMax-H3/diffusers
PROJ=$(cd "$(dirname "$0")" && pwd)
W=${IFAB_WORKSPACE_ROOT:-$PWD}
[ -w "$W" ] || W=/dev/shm/mmh3
mkdir -p "$W/logs" "$W/out" "$H/logs" "$H/output2"
cd "$W" || exit 12
TS=$(date -u +%m%d_%H%M%S)
echo "GPU_COUNT=$(python3 -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null)"
echo "--- setup (cap 900s)"
timeout 900 python3 -u "$PROJ/setup.py" "$W" "$AFS" "$SH" > "$W/logs/nn34_setup.log" 2>&1
SE=$?
echo "SETUP_EXIT=$SE setup_secs=$(( $(date +%s)-T_START ))"
[ "$SE" = "0" ] || { echo "SETUP_FAILED"; tail -10 "$W/logs/nn34_setup.log" | cut -c1-190; echo "=== nn34 done $(date -u +%FT%T)"; exit 0; }
R="$W/InfiniCore/build/linux/x86_64/release"; L="$W/InfiniCore/python/infinicore/lib"
mkdir -p "$L"
cp -f "$R"/_infinicore*.so "$R"/libinfiniop.so "$R"/libinfinicore_cpp_api.so "$R"/libinfinirt.so "$R"/libinfiniccl.so "$L"/ 2>/dev/null
if [ -f "$B/out_release/libinfinicore_cpp_api.so" ]; then
  cp -f "$B/out_release/libinfinicore_cpp_api.so" "$L/libinfinicore_cpp_api.so"
fi
export PYTHONPATH="$W/InfiniCore/python:$W/InfiniLM/python:$W/pylibs"
export LD_LIBRARY_PATH="$L:$W/.infini/lib:$W/fa:$R:/opt/dtk/lib:/opt/dtk/llvm/lib:/opt/hyhal/lib:/opt/dtk/.hyhal/rocm_smi:/opt/dtk/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}"
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
[ -f "$PRE" ] && { export LD_PRELOAD="$PRE:${LD_PRELOAD:-}"; echo "PRELOAD_OK"; }
export HF_HOME="$W/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FA_SO_PATH="$(cat "$W/fa_so_path.txt" 2>/dev/null)"
export PYTHONFAULTHANDLER=1
TP_IDS=${TP_IDS:-0,1,2,3,4,5,6,7}
cd "$W/InfiniLM" || exit 13
run_one () {
  local TAG=$1; shift
  local LOG="$W/logs/nn34_$TAG.log"
  local OUT="$W/out/$TAG.mp4"
  local LEFT=$(( BUDGET - ( $(date +%s) - T_START ) ))
  echo "--- run $TAG left=${LEFT}s"
  local T0=$(date +%s)
  timeout $LEFT python3 -u examples/minimax_h3.py fl2va --model-path "$W/model" \
    --text "A dog falls asleep" --first-frame "$H/input/1280X1280.JPEG" \
    --gpus "$TP_IDS" --num-frames 124 --steps $STEPS --seed 42 --output "$OUT" "$@" > "$LOG" 2>&1
  echo "RUN_$TAG RC=$? secs=$(( $(date +%s)-T0 ))"
  tr '\r' '\n' < "$LOG" | grep -v Materializing | grep -E "Error|error|Traceback|it/s|s/it" | tail -6 | cut -c1-200
  if [ -f "$OUT" ]; then
    cp -f "$OUT" "$H/output2/${TAG}_${TS}.mp4"
    echo "MP4_$TAG bytes=$(stat -c%s "$OUT")"
  else
    echo "NO_MP4_$TAG"
  fi
}
run_one nl_$STEPS
run_one leg_$STEPS --legacy --legacy-text-gpus 0,1,2,3 --legacy-transformer-gpus 4,5,6,7
export PYTHONPATH="$W/pylibs:$H/pylibs_extra:$W/InfiniCore/python:$W/InfiniLM/python"
PERSERIES="nl_$STEPS|leg_$STEPS" timeout 600 python3 -u "$PROJ/probe.py" "$W" 2>&1 | grep -v -E "UserWarning|uniform_filter"
echo "TOTAL_SECS=$(( $(date +%s)-T_START ))"
echo "=== nn34 done $(date -u +%FT%T)"
