#!/bin/bash
# repro_build.sh -- 作业A：从零重建五个 .so（不跑模型，所以只需要 CPU）。
# 达不到「一个作业内做完」的时限要求时，这一步单独成作业，产物落在
#   $HOME/minimax_h3/chain/<UTC 时间戳>/{libs,work,MANIFEST.txt}
# 下一个作业只读使用（AFS 上跨作业只可读、不可写，所以这里收尾 chmod -R a+rwX）。
set -uo pipefail
H=${IFAB_AFS_HOME:-$HOME}; TS=$(date -u +%Y%m%d_%H%M%S)
CH=${CHAIN_DIR:-$H/minimax_h3/chain/$TS}
mkdir -p "$CH" || { echo "CHAIN_DIR_FAIL $CH"; exit 1; }
cd "$CH" || exit 1
echo "JOB_START $(date -u +%FT%T) nproc=$(nproc) uid=$(id -u) chain=$CH"
# 取仓库：tarball 直下优先（快且不依赖 git 的历史），三镜像；git clone 兜底。
fetch_repo() {
  REL=https://github.com/Wang-Yi-Dao/minimax-h3-fl2va-infini-ninetoothed/archive/refs/heads/main.tar.gz
  for pre in https://ghproxy.net/ https://gh-proxy.com/ ""; do
    t0=$(date +%s)
    if curl -fsSL --max-time 100 -o repo.tgz "${pre}${REL}" 2>/dev/null && [ -s repo.tgz ]; then
      tar xzf repo.tgz 2>/dev/null
      d=$(ls -d minimax-h3-fl2va-infini-ninetoothed-* 2>/dev/null | grep -v tar.gz | head -1)
      [ -n "$d" ] && { echo "REPO_TGZ src=${pre:-direct} secs=$(( $(date +%s)-t0 )) dir=$d" >&2; echo "$d"; return 0; }
    fi
    echo "REPO_TRY_FAIL ${pre:-direct} secs=$(( $(date +%s)-t0 ))" >&2
  done
  for u in https://ghproxy.net/https://github.com/Wang-Yi-Dao/minimax-h3-fl2va-infini-ninetoothed.git https://github.com/Wang-Yi-Dao/minimax-h3-fl2va-infini-ninetoothed.git https://gitclone.com/github.com/Wang-Yi-Dao/minimax-h3-fl2va-infini-ninetoothed.git; do
    t0=$(date +%s)
    if timeout 240 git clone --depth 1 -q "$u" repo 2>/dev/null; then
      echo "REPO_CLONE src=$u secs=$(( $(date +%s)-t0 ))" >&2
      echo repo; return 0
    fi
    echo "REPO_CLONE_FAIL $u" >&2
  done
  return 1
}
D=$(fetch_repo) || { echo "REPO_FAIL"; echo "JOB_END $(date -u +%FT%T)"; exit 1; }
cd "$CH/$D" || exit 1
echo "--- repo ready, inputs:"; ls -l src | awk '{print "  IN",$5,$9}'
( cd src && sha256sum -c --strict SHA256SUMS 2>&1 | cut -c1-60 ) | sed 's/^/SHA /'
mkdir -p "$CH/work"
echo "--- build start $(date -u +%FT%T)"
PY=${PY:-python3.11} J=${J:-64} ARCH=${ARCH:-gfx936} \
  bash build/reproduce.sh "$CH/work" all
BRC=$?
echo "REPRO_EXIT=$BRC"
FR=$CH/work/ic/build/linux/x86_64/release
mkdir -p "$CH/libs"
N=0
for f in _infinicore.cpython-311-x86_64-linux-gnu.so libinfiniop.so libinfinicore_cpp_api.so libinfinirt.so libinfiniccl.so; do
  if [ -s "$FR/$f" ]; then cp -f "$FR/$f" "$CH/libs/"; N=$((N+1)); echo "SO $f $(stat -c%s "$CH/libs/$f")"; fi
done
echo "SO_COUNT=$N/5"
{
  echo "chain $CH"
  echo "created $(date -u +%FT%T)"
  echo "repo src=github main ($( cd "$CH/$D" && git rev-parse HEAD 2>/dev/null || echo tarball ))"
  echo "reproduce_exit $BRC  so_count $N"
  ( cd "$CH/libs" && sha256sum * 2>/dev/null )
} > "$CH/MANIFEST.txt" 2>&1
cat "$CH/MANIFEST.txt" | sed 's/^/MAN /'
chmod -R a+rwX "$CH" 2>/dev/null || echo "CHMOD_WARN (not owner?)"
if [ "$BRC" = 0 ] && [ "$N" = 5 ]; then echo "CHAIN_OK $CH"; else echo "CHAIN_FAIL brc=$BRC so=$N"; fi
echo "JOB_END $(date -u +%FT%T)"