#!/bin/bash
# nn36: (A) 九齿 FA 在真实长序列 S=27570 下与 torch SDPA 的误差; (B) 全模型 TP8 2 步改用 ATen attention 出片，与 legacy 2 步对比
set +u
STEPS=${1:-${STEPS:-2}}
BUDGET=${2:-${BUDGET:-1500}}
T_START=$(date +%s)
echo "=== nn36 start $(date -u +%FT%T) steps=$STEPS budget=$BUDGET"
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
timeout 900 python3 -u "$PROJ/setup.py" "$W" "$AFS" "$SH" > "$W/logs/nn36_setup.log" 2>&1
SE=$?
echo "SETUP_EXIT=$SE setup_secs=$(( $(date +%s)-T_START ))"
[ "$SE" = "0" ] || { echo "SETUP_FAILED"; tail -10 "$W/logs/nn36_setup.log" | cut -c1-190; echo "=== nn36 done $(date -u +%FT%T)"; exit 0; }
R="$W/InfiniCore/build/linux/x86_64/release"; L="$W/InfiniCore/python/infinicore/lib"
mkdir -p "$L"
cp -f "$R"/_infinicore*.so "$R"/libinfiniop.so "$R"/libinfinicore_cpp_api.so "$R"/libinfinirt.so "$R"/libinfiniccl.so "$L"/ 2>/dev/null
if [ -f "$B/out_release/libinfinicore_cpp_api.so" ]; then
  cp -f "$B/out_release/libinfinicore_cpp_api.so" "$L/libinfinicore_cpp_api.so"
fi
export PYTHONPATH="$W/InfiniCore/python:$W/InfiniLM/python:$W/pylibs"
export LD_LIBRARY_PATH="$L:$W/.infini/lib:$W/fa:$R:/opt/dtk/lib:/opt/dtk/llvm/lib:/opt/hyhal/lib:/opt/dtk/.hyhal/rocm_smi:/opt/dtk/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}"
PRE=$B/pre_ctxid.so
if [ -f "$PRE" ]; then
  export LD_PRELOAD="$PRE:${LD_PRELOAD:-}"
  echo "PRELOAD_OK"
fi
export HF_HOME="$W/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FA_SO_PATH="$(cat "$W/fa_so_path.txt" 2>/dev/null)"
export PYTHONFAULTHANDLER=1
echo "--- part A: FA accuracy at long S"
timeout 600 python3 -u "$PROJ/fa.py" 2>&1 | grep -E "^FA_S|^NN36FA|Error|Traceback" | cut -c1-220
echo "--- part B: full model TP8 steps=$STEPS with ATen attention"
cd "$W/InfiniLM" || exit 13
export H3_FORCE_ATEN_ATTN=1
OUT=$W/out/nl${STEPS}_aten.mp4
LOG=$W/logs/nn36_aten.log
LEFT=$(( BUDGET - ( $(date +%s) - T_START ) ))
echo "run nl${STEPS}_aten left=${LEFT}s"
T0=$(date +%s)
timeout $LEFT python3 -u examples/minimax_h3.py fl2va --model-path "$W/model" \
  --text "A dog falls asleep" --first-frame "$H/input/1280X1280.JPEG" \
  --gpus 0,1,2,3,4,5,6,7 --num-frames 124 --steps $STEPS --seed 42 --output "$OUT" > "$LOG" 2>&1
echo "RUN_aten RC=$? secs=$(( $(date +%s)-T0 ))"
tr '\r' '\n' < "$LOG" | grep -v Materializing | grep -E "Error|error|Traceback|it/s|s/it|out of memory" | tail -8 | cut -c1-220
if [ -f "$OUT" ]; then
  cp -f "$OUT" "$H/output2/nl${STEPS}_aten_$TS.mp4"
  echo "MP4_aten bytes=$(stat -c%s "$OUT")"
else
  echo "NO_MP4_aten"
fi
export PYTHONPATH="$W/pylibs:$H/pylibs_extra:$W/InfiniCore/python:$W/InfiniLM/python"
PERSERIES="nl${STEPS}_aten|leg_$STEPS" timeout 600 python3 -u "$PROJ/probe.py" "$W" 2>&1 | grep -v -E "UserWarning|uniform_filter" | grep -E "VIDS|STAT|SSIM|PSER|frames|ssim|PROBE3" | cut -c1-260
echo "TOTAL_SECS=$(( $(date +%s)-T_START ))"
echo "=== nn36 done $(date -u +%FT%T)"
