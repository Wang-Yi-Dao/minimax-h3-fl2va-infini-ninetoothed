#!/bin/bash
# repro_run.sh -- 作业B：用作业A重建出的五个 .so 跑 FL2VA 50 步并执行官方 compare。
# 只读使用 chain 目录（AFS 跨作业只可读），不复制、不修改上游产物。
#   usage: repro_run.sh [steps] [budget]
set -uo pipefail
STEPS=${1:-50}; BUDGET=${2:-1500}
H=${IFAB_AFS_HOME:-$HOME}
if [ -n "${CHAIN_DIR:-}" ]; then
  CH=$CHAIN_DIR
else
  # 取链上最新的、且确实含 libs 的那一个（时间戳目录名按字典序即时间序）
  CH=""
  for d in $(ls -d $H/minimax_h3/chain/*/ 2>/dev/null | sort -r); do
    if [ -d "${d}libs" ]; then CH=${d%/}; break; fi
  done
fi
CH=${CH%/}
echo "RUN_START $(date -u +%FT%T) steps=$STEPS budget=$BUDGET chain=$CH"
[ -d "$CH/libs" ] || { echo "CHAIN_LIBS_MISSING $CH/libs"; exit 1; }
for f in _infinicore.cpython-311-x86_64-linux-gnu.so libinfiniop.so libinfinicore_cpp_api.so libinfinirt.so libinfiniccl.so; do
  s=$(stat -c%s "$CH/libs/$f" 2>/dev/null || echo 0)
  echo "CHAIN_SO $f $s"
  [ "$s" != 0 ] || { echo "CHAIN_SO_MISSING $f"; exit 1; }
done
cd $H/minimax_h3/tmp 2>/dev/null || cd /tmp
rm -rf rrun; mkdir -p rrun; cd rrun || exit 1
fetch_repo() {
  REL=https://github.com/Wang-Yi-Dao/minimax-h3-fl2va-infini-ninetoothed/archive/refs/heads/main.tar.gz
  for pre in https://ghproxy.net/ https://gh-proxy.com/ ""; do
    t0=$(date +%s)
    if curl -fsSL --max-time 100 -o repo.tgz "${pre}${REL}" 2>/dev/null && [ -s repo.tgz ]; then
      tar xzf repo.tgz 2>/dev/null
      d=$(ls -d minimax-h3-fl2va-infini-ninetoothed-* 2>/dev/null | grep -v tar.gz | head -1)
      [ -n "$d" ] && { echo "REPO_TGZ src=${pre:-direct} secs=$(( $(date +%s)-t0 ))" >&2; echo "$d"; return 0; }
    fi
    echo "REPO_TRY_FAIL ${pre:-direct}" >&2
  done
  for u in https://ghproxy.net/https://github.com/Wang-Yi-Dao/minimax-h3-fl2va-infini-ninetoothed.git https://github.com/Wang-Yi-Dao/minimax-h3-fl2va-infini-ninetoothed.git; do
    if timeout 240 git clone --depth 1 -q "$u" repo 2>/dev/null; then echo "REPO_CLONE $u" >&2; echo repo; return 0; fi
  done
  return 1
}
D=$(fetch_repo) || { echo REPO_FAIL; exit 1; }
cd "$D" || exit 1
echo "--- repo ok, CRB_LIBS=$CH/libs"
CRB_LIBS="$CH/libs" STEPS=$STEPS BUDGET=$BUDGET bash jobs/run_tp8_fl2va.sh "$STEPS" "$BUDGET" 2>&1 | tail -80
echo "--- compare lines (logs written during this run only)"
# 只取本次作业运行窗口内写过的日志（-2h），避免把上一次运行的旧 cmp 日志也 grep 进来
NEW_LOGS=$(find $H/minimax_h3/logs -maxdepth 1 -name "*.log" -newermt "-2 hours" 2>/dev/null | head -5)
[ -n "$NEW_LOGS" ] && grep -aE "Size:|Length:|Average SSIM|CMP_RC" $NEW_LOGS 2>/dev/null | tail -8
echo "JOB_END $(date -u +%FT%T)"