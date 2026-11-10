#!/bin/bash
# repro_deps.sh -- 只做工具链（作业链第 1 步）：编/取 xmake 并打包，交给下一步。
# 优先用上游官方 bundle（xmake-bundle-v3.1.2.linux.x86_64），跑不起来才源码编译。
set -uo pipefail
H=${IFAB_AFS_HOME:-$HOME}; TS=$(date -u +%Y%m%d_%H%M%S)
CH=${CHAIN_DIR:-$H/minimax_h3/chain/$TS}
mkdir -p "$CH" && cd "$CH" || exit 1
echo "JOB_START $(date -u +%FT%T) nproc=$(nproc) chain=$CH"
fetch_repo() {
  REL=https://github.com/Wang-Yi-Dao/minimax-h3-fl2va-infini-ninetoothed/archive/refs/heads/main.tar.gz
  for pre in https://ghproxy.net/ https://gh-proxy.com/ ""; do
    if curl -fsSL --max-time 100 -o repo.tgz "${pre}${REL}" 2>/dev/null && [ -s repo.tgz ]; then
      tar xzf repo.tgz 2>/dev/null
      d=$(ls -d minimax-h3-fl2va-infini-ninetoothed-* 2>/dev/null | grep -v tar.gz | head -1)
      [ -n "$d" ] && { echo "REPO_TGZ src=${pre:-direct}" >&2; echo "$d"; return 0; }
    fi
    echo "REPO_TRY_FAIL ${pre:-direct}" >&2
  done
  for u in https://ghproxy.net/https://github.com/Wang-Yi-Dao/minimax-h3-fl2va-infini-ninetoothed.git https://github.com/Wang-Yi-Dao/minimax-h3-fl2va-infini-ninetoothed.git; do
    if timeout 240 git clone --depth 1 -q "$u" repo 2>/dev/null; then echo "REPO_CLONE $u" >&2; echo repo; return 0; fi
  done
  return 1
}
D=$(fetch_repo) || { echo REPO_FAIL; exit 1; }
cd "$CH/$D" || exit 1
echo "--- deps start $(date -u +%FT%T)"
XM_OUT="$CH/xmake-built.tgz" PY=${PY:-python3} J=${J:-64} \
  bash build/reproduce.sh "$CH/work" deps
RC=$?
echo "DEPS_EXIT=$RC"
ls -l "$CH/xmake-built.tgz" 2>/dev/null | awk '{print "XM_TGZ_OUT",$5}'
sha256sum "$CH/xmake-built.tgz" 2>/dev/null | sed 's/^/XM_SHA /'
chmod -R a+rwX "$CH" 2>/dev/null || echo CHMOD_WARN
echo "CHAIN_DEPS $CH"
echo "JOB_END $(date -u +%FT%T)"