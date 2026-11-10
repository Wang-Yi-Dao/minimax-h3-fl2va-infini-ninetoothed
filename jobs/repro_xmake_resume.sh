#!/bin/bash
# repro_xmake_resume.sh -- 接着上一次作业编到一半的 xmake 继续编。
#
# 为什么要单独一个脚本：作业有硬时限（3200 s），xmake 的 make 曾经在时限内没编完。
# 上一个作业虽然被杀，但它编出来的 .o 文件留在 AFS 上（属主是别的 uid，只能读）。
# make 是可续的：把这些文件按新属主复制一份，再跑 make，只会补编剩下的部分。
#
#   SRCM  work/xm 的源目录（默认：chain/ 下最新的、确实含 .o 的那个）
#   J     make 并行度（默认 64；节点 128 核，之前用 32 明显没吃满）
set -uo pipefail
H=$HOME/minimax_h3
J=${J:-64}
TS=$(date -u +%Y%m%d_%H%M%S)
CH=$H/chain/$TS
mkdir -p "$CH/logs" "$CH/xm"

echo "RESUME_START $(date -u +%FT%T) j=$J chain=$CH"

# ---- 1. 找上一个作业留下的 xmake 源码树（要有 build 产物，才值得续）----
# 候选目录由 find 扫出（不用 glob：glob 在只有一种布局时会漏掉另一种），
# 每个候选数一遍 .o，挑最多的那个 —— 编得最多的树最值得续。
SRCM=${SRCM:-}
if [ -z "$SRCM" ]; then
  BEST=0
  for d in $(find "$H/chain" -maxdepth 4 -type d -name xm 2>/dev/null | sort -r); do
    N=$(find "$d" -name '*.o' 2>/dev/null | wc -l)
    echo "RESUME_PROBE $d objs=$N"
    if [ "$N" -gt "$BEST" ]; then BEST=$N; SRCM=$d; fi
  done
  echo "RESUME_PICK ${SRCM:-none} objs=$BEST"
fi
if [ -z "$SRCM" ] || [ ! -d "$SRCM" ]; then
  echo "RESUME_NO_SRC 找不到含 .o 的 xmake 源码树"; exit 1
fi
NOBJ=$(find "$SRCM" -name '*.o' 2>/dev/null | wc -l)
NSRC=$(find "$SRCM" -name '*.c' 2>/dev/null | wc -l)
echo "RESUME_SRC $SRCM objs=$NOBJ csources=$NSRC"

# ---- 2. 复制成自己属主（跨作业只能读别人的文件，复制后就归自己）----
T0=$(date +%s)
cp -a --no-preserve=ownership "$SRCM"/. "$CH/xm"/ 2>/dev/null || cp -r "$SRCM"/. "$CH/xm"/
echo "RESUME_COPY $(( $(date +%s) - T0 ))s files=$(find "$CH/xm" -type f | wc -l)"

# ---- 3. 找 configure 所在目录，增量 make ----
XSRC=$(find "$CH/xm" -maxdepth 4 -type f -name configure -perm -u+x 2>/dev/null | head -1)
[ -n "$XSRC" ] || { echo "RESUME_NO_CONFIGURE"; exit 1; }
XSRC=$(dirname "$XSRC")
# 复制过来的树若缺 Makefile（configure 没跑完就被杀），先补跑 configure。
if [ ! -f "$XSRC/Makefile" ]; then
  echo "RESUME_CONFIGURE $(date -u +%FT%T)"
  ( cd "$XSRC" && ./configure ) > "$CH/logs/configure.log" 2>&1 || { echo RESUME_CONFIGURE_FAIL; tail -10 "$CH/logs/configure.log" | cut -c1-160; exit 1; }
fi
echo "RESUME_MAKE_START $(date -u +%FT%T) src=$XSRC"
( cd "$XSRC" && make -j"$J" ) > "$CH/logs/make.log" 2>&1 &
MPID=$!
while kill -0 "$MPID" 2>/dev/null; do
  sleep 60
  echo "MAKE $(date -u +%T) lines=$(wc -l < "$CH/logs/make.log" 2>/dev/null) err=$(grep -c 'error:' "$CH/logs/make.log" 2>/dev/null)"
done
wait "$MPID"; MRC=$?
echo "MAKE_RC=$MRC lines=$(wc -l < "$CH/logs/make.log")"
tail -5 "$CH/logs/make.log" | cut -c1-160

# ---- 4. 打包交给下一个作业 ----
XMBIN=$(find "$CH/xm" -maxdepth 6 -type f -name xmake -perm -u+x 2>/dev/null | head -1)
if [ -z "$XMBIN" ]; then echo "RESUME_NO_BIN rc=$MRC"; exit 1; fi
echo "XM_BIN $XMBIN $("$XMBIN" --version 2>&1 | head -1)"
OUT=$CH/xmake-built.tgz
tar czf "$OUT" -C "$CH" xm || { echo RESUME_TAR_FAIL; exit 1; }
chmod -R a+rwX "$CH" 2>/dev/null
{ echo "out=$OUT"; echo "bytes=$(stat -c%s "$OUT")"; echo "objs=$NOBJ"; echo "sha256=$(sha256sum "$OUT" | cut -d' ' -f1)"; } > "$CH/MANIFEST.txt"
echo "RESUME_OK $OUT $(stat -c%s "$OUT") B"
echo "JOB_END $(date -u +%FT%T)"
