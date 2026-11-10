#!/bin/bash
# reproduce.sh -- 从公开仓库一条命令重建五个 .so（报告 9.3 节的重建步骤）
#
#   git clone https://github.com/Wang-Yi-Dao/minimax-h3-fl2va-infini-ninetoothed.git
#   cd minimax-h3-fl2va-infini-ninetoothed
#   bash build/reproduce.sh                 # 工作目录默认本地盘 /tmp/repro-build
#   bash build/reproduce.sh /data/bw        # 或显式指定
#
# 需要：python3.11、DTK（/opt/dtk，带 hipcc 与 clang）、能访问 GitHub 与 PyPI。
# 三份源码输入随包在 src/ 下，脚本先按 src/SHA256SUMS 校验再构建；
# 构建依赖（xmake 3.1.2 源码树、pybind11 3.1.0 头文件）由脚本自己下载，
# 下载失败可用 GH_MIRROR 换镜像，或预置 XM_TGZ / PYBIND11_ROOT。
# 工作目录按顺序探活：$LOCAL_SCRATCH、/tmp、$HOME/tmp/repro-build，第一个能写的就用。
# 快照解包本身不慢：本集群实测 916 个小文件 9 秒、7469 个成员约 40-75 秒，
# 真正的耗时是编 xmake 与编 InfiniCore，所以工具链优先用官方预编译 bundle。
# 产物：<work>/ic/build/linux/x86_64/release 下五个 .so。
# 端到端出片 + 官方评分见 build/README.md（jobs/run_tp8_fl2va.sh，另需模型权重与参考视频）。
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
# 工作目录：优先本地盘（快），没有可写本地盘就退回持久盘。
# 这套集群实测 /dev/shm、/tmp、/var/tmp 虽然是 1777 权限，mkdir 仍被拒，
# 只有 $HOME 可写，所以下面按顺序探活（真的写一个文件），第一个能写的就用。
pick_work() {
  for d in ${LOCAL_SCRATCH:-/dev/shm/repro-build} /tmp/repro-build "$HOME/tmp/repro-build"; do
    mkdir -p "$d" 2>/dev/null || continue
    touch "$d/.probe" 2>/dev/null || continue
    rm -f "$d/.probe"; echo "$d"; return 0
  done
  return 1
}
if [ -n "${1:-}" ]; then W="$1"; else W=$(pick_work) || { echo "WORKDIR_NONE"; exit 1; }; fi
# 阶段：all = 一次做完（慢机器可能超出单作业时限）；deps = 只做依赖与 xmake
# 工具链并打包；build = 用 ${XM_PREBUILT} 里已编好的 xmake 只编 InfiniCore。
STAGE=${2:-all}
PY=${PY:-python3.11}
DTK=${DTK:-/opt/dtk}
J=${J:-32}
ARCH=${ARCH:-gfx936}
MIR=${GH_MIRROR:-https://ghproxy.net/}
DEPS=${DEPS:-$ROOT/.deps}
SRC=$ROOT/src
T0=$(date +%s)

mkdir -p "$W" 2>/dev/null
if ! touch "$W/.w" 2>/dev/null; then echo "WORKDIR_NOT_WRITABLE $W"; exit 1; fi
rm -f "$W/.w"
echo "REPRO_START $(date -u +%FT%T) root=$ROOT work=$W"
echo "PY=$PY $("$PY" -V 2>&1) DTK=$DTK J=$J ARCH=$ARCH"

# ---- 1. 三份源码输入：存在性 + sha256 ----
for f in InfiniCore-snapshot-mit.tar.gz overlay.tar.gz kernels_deployed.tgz; do
  [ -s "$SRC/$f" ] || { echo "MISSING_INPUT $SRC/$f"; exit 1; }
done
if ( cd "$SRC" && sha256sum -c --strict SHA256SUMS ); then
  echo "SHA_OK all three inputs"
else
  echo "SHA_FAIL"; exit 1
fi

# ---- 2. 构建依赖：xmake 3.1.2 源码树 + pybind11 3.1.0 ----
mkdir -p "$DEPS"
XM_TGZ=${XM_TGZ:-$DEPS/xmake-v3.1.2.tar.gz}
if [ ! -s "$XM_TGZ" ]; then
  echo "--- download xmake 3.1.2 source tree"
  XM_TGZ="$XM_TGZ" GH_MIRROR="$MIR" "$PY" - <<'PYEOF' || { echo "XMAKE_DL_FAIL"; exit 1; }
import os, sys, urllib.request
dst = os.environ["XM_TGZ"]
url = "https://github.com/xmake-io/xmake/releases/download/v3.1.2/xmake-v3.1.2.tar.gz"
mir = os.environ.get("GH_MIRROR", "https://ghproxy.net/") + url
for u in (url, mir):
    try:
        with urllib.request.urlopen(u, timeout=180) as r:
            data = r.read()
        if len(data) < 1000000:
            raise RuntimeError("short body %d" % len(data))
        open(dst, "wb").write(data)
        print("XMAKE_TGZ_BYTES", len(data), u)
        break
    except Exception as exc:
        print("XMAKE_DL_FAIL", u, exc)
else:
    sys.exit(1)
PYEOF
fi

# ---- 2b. 只做工具链：编 xmake 并打包，交给下一个作业用 ----
# 放在 pybind11 之前：工具链阶段用不到 pybind11，早退能省掉一次下载与解包。
if [ "$STAGE" = deps ]; then
  OUT=${XM_OUT:-$DEPS/xmake-built.tgz}
  W="$W" XM_TGZ="$XM_TGZ" OUT="$OUT" J="$J" bash "$HERE/xmake_build.sh" || { echo "DEPS_FAIL"; exit 1; }
  echo "DEPS_OK xmake_prebuilt=$OUT $(stat -c%s "$OUT" 2>/dev/null) B"
  echo "next: 作业2 用 XM_PREBUILT=$OUT bash build/reproduce.sh <work2> build"
  exit 0
fi

PB=${PYBIND11_ROOT:-$DEPS/pybind11-3.1.0}
if [ ! -f "$PB/include/pybind11/pybind11.h" ]; then
  echo "--- download pybind11 3.1.0 source"
  PYBIND11_ROOT="$PB" "$PY" - <<'PYEOF' || { echo "PYBIND11_DL_FAIL"; exit 1; }
import io, json, os, sys, tarfile, urllib.request
dst = os.environ["PYBIND11_ROOT"]
meta = json.load(urllib.request.urlopen("https://pypi.org/pypi/pybind11/3.1.0/json", timeout=120))
url = next(f["url"] for f in meta["urls"] if f["filename"].endswith(".tar.gz"))
print("PYBIND11_SDIST", url)
raw = urllib.request.urlopen(url, timeout=300).read()
os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(raw)) as t:
    t.extractall(os.path.dirname(os.path.abspath(dst)) or ".", filter="data")
print("PYBIND11_ROOT", dst, "hdr", os.path.isfile(os.path.join(dst, "include", "pybind11", "pybind11.h")))
PYEOF
fi
echo "XMAKE_TGZ=$(stat -c%s "$XM_TGZ" 2>/dev/null) PYBIND11_ROOT=$PB"

# ---- 3. 六步构建（build/from_scratch.sh）----
mkdir -p "$W" "$W/scratch/tmp"
export W PY DTK J ARCH PHASE=all
export IC_TGZ=$SRC/InfiniCore-snapshot-mit.tar.gz
export OVL_TGZ=$SRC/overlay.tar.gz
export CRB_KERNELS_TGZ=$SRC/kernels_deployed.tgz
export XMAKE_TGZ="$XM_TGZ" CRB_SCRATCH=$W/scratch PYBIND11_ROOT="$PB"
# CRB_RESUME=1: reuse the source tree and xmake build artifacts left by an
# earlier run in the same work directory. A single job has a hard deadline
# (3200 s on this platform); on a slow filesystem use this to continue.
export CRB_RESUME=${CRB_RESUME:-0}
# build 阶段必须有已编好的 xmake：没有它 from_scratch.sh 会退回源码编译，
# 而这一步在慢速持久盘上要 30 分钟以上，会直接撞到作业时限。
if [ "$STAGE" = build ]; then
  XM_PREBUILT=${XM_PREBUILT:-$DEPS/xmake-built.tgz}
  [ -s "$XM_PREBUILT" ] || { echo "XM_PREBUILT_MISSING $XM_PREBUILT"; exit 1; }
  export XM_PREBUILT
  echo "prebuilt xmake: $XM_PREBUILT $(stat -c%s "$XM_PREBUILT") B"
fi
# WHEELS 可指向本地轮子目录以跳过下载；不设则由 from_scratch.sh 去 PyPI 取。
[ -n "${WHEELS:-}" ] && export WHEELS
LOG=$W/from_scratch.log
bash "$HERE/from_scratch.sh" > "$LOG" 2>&1 &
PYID=$!
LASTR=0
while kill -0 "$PYID" 2>/dev/null; do
  sleep 15
  NOW=$(date +%s)
  if [ $((NOW - LASTR)) -ge 90 ]; then
    LASTR=$NOW
    echo "--- $(date -u +%T) build $(wc -l < "$LOG") lines $(grep -c 'error:' "$LOG" 2>/dev/null) err: $(grep -aE '^=== \[' "$LOG" | tail -1 | cut -c1-70)"
  fi
done
wait "$PYID"; BR=$?
echo "BUILD_EXIT=$BR"
grep -aE "OVL_FILES_WRITTEN|KERNEL_FILES|DISPATCH|PATCH .* (OK|MISS)|BIND_RC|error:" "$LOG" | tail -10 | cut -c1-180

# ---- 4. 产物核对 ----
FR=$W/ic/build/linux/x86_64/release
OVER=0
for f in _infinicore.cpython-311-x86_64-linux-gnu.so libinfiniop.so libinfinicore_cpp_api.so libinfinirt.so libinfiniccl.so; do
  if [ -s "$FR/$f" ]; then OVER=$((OVER + 1)); echo "SO $f $(stat -c%s "$FR/$f")"; else echo "MISSING $f"; fi
done
echo "SO_COUNT=$OVER/5"
if [ "$BR" != "0" ] || [ "$OVER" != "5" ]; then
  echo "REPRO_FAIL build_exit=$BR so_count=$OVER"
  tail -20 "$LOG" | cut -c1-180
  echo "--- full log: $LOG"
  exit 1
fi
echo "REPRO_OK secs=$(( $(date +%s) - T0 )) libs=$FR"
echo "next (end-to-end + official compare): CRB_LIBS=$FR bash jobs/run_tp8_fl2va.sh 50 1600"
