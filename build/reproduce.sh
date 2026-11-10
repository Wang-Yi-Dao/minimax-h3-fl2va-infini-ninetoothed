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
# 工作目录默认落在本地盘：上游快照 7.5k 个小文件解在 AFS 上要 19 分钟，
# 本地盘几十秒；构建结束后把 5 个 .so 拷到想要的位置即可。
# 产物：<work>/ic/build/linux/x86_64/release 下五个 .so。
# 端到端出片 + 官方评分见 build/README.md（jobs/run_tp8_fl2va.sh，另需模型权重与参考视频）。
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
LOCAL=${LOCAL_SCRATCH:-/dev/shm/repro-build}
W=${1:-$LOCAL}
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
