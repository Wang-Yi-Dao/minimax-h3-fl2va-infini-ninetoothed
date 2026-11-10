#!/bin/bash
# from_scratch.sh -- rebuild the InfiniCore / nine-toothed side of the FL2VA
# pipeline from a pinned upstream snapshot plus this package's overlay files.
#
#   IC_TGZ       upstream InfiniCore snapshot tarball (top dir InfiniCore/)
#   OVL_TGZ      overlay tarball from this package (patch/overlay.tar.gz)
#   XMAKE_TGZ    xmake 3.1.2 release tarball -- upstream publishes no portable
#                linux binary (the bundle needs GLIBC_2.38), so this is the source
#                tree and step 0 builds it in place
#   WHEELS       directory holding triton / ninetoothed / numpy wheels (optional)
#   PYBIND11_ROOT PYTHON_INCLUDE PYTHON_LIBDIR PYTHON_LIBNAME  build deps
#   W            work directory (default ./build-work)   DTK  DTK root (/opt/dtk)
#
# Every non-upstream change is applied by a small, named step below, so the
# result can be diffed against the upstream tree to see exactly what changed.
set -uo pipefail
stage() { echo "=== [$1] $(date -u +%FT%T)"; }

W=${W:-$PWD/build-work}
DTK=${DTK:-/opt/dtk}
IC_TGZ=${IC_TGZ:?set IC_TGZ}
OVL_TGZ=${OVL_TGZ:?set OVL_TGZ}
XMAKE_TGZ=${XMAKE_TGZ:?set XMAKE_TGZ}
WHEELS=${WHEELS:-}
PYBIND11_ROOT=${PYBIND11_ROOT:?set PYBIND11_ROOT}
PYTHON_INCLUDE=${PYTHON_INCLUDE:-/usr/local/include/python3.11}
PYTHON_LIBDIR=${PYTHON_LIBDIR:-/usr/local/lib}
PYTHON_LIBNAME=${PYTHON_LIBNAME:-python3.11}
# The interpreter must be the one the wheels were built for: the DTK torch is
# cp311-only, so a 3.11 wheel set is what actually works here (see build/README.md).
PY=${PY:-python3.11}
J=${J:-32}
ARCH=${ARCH:-gfx936}

XM_BIN=$(find "$W/xm" -maxdepth 5 -type f -name xmake -perm -u+x 2>/dev/null | head -1)
XM_SHARE=$(dirname "$(dirname "$XM_BIN")")/share/xmake
PHASE=${PHASE:-all}      # prep = fetch/patch/generate kernels, build = compile only, all = both
do_prep=1; [ "$PHASE" = "build" ] && do_prep=0

if [ "$do_prep" = 1 ]; then
stage "prep (work=$W)"
mkdir -p "$W"/py "$W"/ic "$W"/xm "$W"/tmp "$W"/tcache "$W"/logs
WHL_PID=
if [ -z "$WHEELS" ]; then
  # No wheel directory given: fetch the four wheels (ninetoothed, sympy, mpmath,
  # numpy, plus the Hygon triton build) from PyPI's JSON API.  pip in this image
  # cannot do it -- see build/README.md and fetch_wheels.py.
  # This is a few hundred MB of downloads, so run it in the background and let the
  # snapshot extraction and the xmake compile below overlap with it; the unpack
  # happens in step 2, once both sides are done.
  WHEELS="$W/whl"
  "$PY" "$(dirname "$0")/fetch_wheels.py" "$WHEELS" > "$W/logs/wheels.log" 2>&1 &
  WHL_PID=$!
  echo "wheels: fetching into $WHEELS in the background (pid $WHL_PID)"
fi
# The upstream snapshot is ~7.5k files, which costs ~19 min on AFS.  Set
# CRB_RESUME=1 to reuse a tree extracted by an earlier run; the overlay (9
# files) is always re-applied so that a changed overlay always wins.
OVL_SHA=$(sha256sum "$OVL_TGZ" | cut -d' ' -f1)
if [ "${CRB_RESUME:-0}" = 1 ] && [ -f "$W/ic/xmake.lua" ]; then
  echo "snapshot already extracted, re-applying overlay $OVL_SHA"
else
  tar xzf "$IC_TGZ" -C "$W/ic" --strip-components=1 || exit 1
fi
# The work tree may have been produced by an earlier run under a different uid
# mapping; GNU tar then refuses with "Cannot open: File exists".  Unpack the nine
# overlay files with python instead, chmod-ing each target first.
"$PY" - "$OVL_TGZ" "$W/ic" <<'PYEOF' || exit 1
import os, sys, tarfile
tgz, dst = sys.argv[1], sys.argv[2]
bad = 0
with tarfile.open(tgz) as t:
    for m in t.getmembers():
        p = os.path.join(dst, m.name)
        if m.isdir():
            os.makedirs(p, exist_ok=True)
            continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        try:
            os.chmod(p, 0o644)
        except OSError:
            pass
        try:
            with open(p, "wb") as fh:
                fh.write(t.extractfile(m).read())
        except OSError as exc:
            print("OVL_WRITE_FAIL", m.name, exc)
            bad += 1
print("OVL_FILES_WRITTEN", 9 - bad, "failed", bad)
sys.exit(1 if bad else 0)
PYEOF
grep -l h3_pad80 "$W/ic/src/infinicore/ops/multi_head_attention/hygon_nt/mha_ninetoothed_hygon.cc" >/dev/null || { echo "OVL_VERIFY_FAIL"; exit 1; }
echo "$OVL_SHA" > "$W/ic/.crb_overlay_sha"
mkdir -p "$W/logs"
if [ -n "${XM_PREBUILT:-}" ] && [ -s "${XM_PREBUILT}" ]; then
  # A toolchain built by an earlier job (see build/xmake_build.sh).  Compiling
  # xmake here would cost ~30 min on this cluster's persistent storage, which
  # does not fit in a single job window, so that one step gets its own job and
  # hands the finished tree forward as one read-only archive.
  rm -rf "$W/xm"; mkdir -p "$W/xm"
  tar xzf "$XM_PREBUILT" -C "$W/xm" || { echo "XM_PREBUILT_UNPACK_FAIL"; exit 1; }
  echo "xmake: unpacked prebuilt toolchain $XM_PREBUILT"
elif [ "${CRB_RESUME:-0}" = 1 ] && [ -n "$XM_BIN" ] && [ -x "$XM_BIN" ]; then
  echo "xmake already extracted, reusing $XM_BIN"
else
  rm -rf "$W/xm"; mkdir -p "$W/xm"
  tar xzf "$XMAKE_TGZ" -C "$W/xm" || exit 1
fi
# xmake reaches these through os.getenv(), so they must live in the environment.
export PYTHON_INCLUDE PYTHON_LIBDIR PYTHON_LIBNAME PYBIND11_ROOT
export J ARCH DTK
# xmake may be extracted by *this* run (fresh work directory), so the binary and
# its module directory have to be located again before they are used.
XM_BIN=$(find "$W/xm" -maxdepth 5 -type f -name xmake -perm -u+x 2>/dev/null | head -1)
if [ -n "$XM_BIN" ]; then
  XM_SHARE=$(dirname "$(dirname "$XM_BIN")")/share/xmake
else
  # The xmake release tarball is a *source* tree: upstream ships no portable
  # linux binary (the bundle needs GLIBC_2.38, this image has 2.35), so build
  # it here.  ~3-4 min; ./configure writes build/linux/x86_64/release, make
  # links build/xmake, and the Lua program tree lives in <src>/xmake.
  XSRC=$(find "$W/xm" -maxdepth 4 -type f -name configure -perm -u+x 2>/dev/null | head -1)
  [ -n "$XSRC" ] || { echo "XMAKE_NOT_FOUND under $W/xm"; exit 1; }
  XSRC=$(dirname "$XSRC")
  stage "step 0/6 xmake from the release source tree"
  ( cd "$XSRC" && ./configure > "$W/logs/xmake-configure.log" 2>&1 && \
    make -j"$J" > "$W/logs/xmake-make.log" 2>&1 ) || {
      echo "XMAKE_BUILD_FAIL"; tail -20 "$W/logs/xmake-make.log" 2>/dev/null | cut -c1-160; exit 1; }
  XM_BIN="$XSRC/build/xmake"
  XM_SHARE="$XSRC/xmake"   # program dir of a source build, not share/xmake
fi
[ -x "$XM_BIN" ] || { echo "XMAKE_NOT_FOUND under $W/xm"; exit 1; }
echo "xmake=$XM_BIN share=$XM_SHARE"

stage "step 1/6 thrust version shim (DTK thrust 2.3.2 vs CCCL 2.4.0 static_assert)"
"$PY" - "$W" <<'PYEOF'
import io, os, sys
W = sys.argv[1]; ic = os.path.join(W, 'ic')
shim = os.path.join(ic, 'xmake', 'thrust_shim', 'thrust')
os.makedirs(shim, exist_ok=True)
io.open(os.path.join(shim, 'version.h'), 'w', encoding='utf-8').write('''#pragma once
/* DTK ships thrust 2.3.2 together with CCCL/CUB 2.4.0; upstream thrust turns
   that skew into a hard static_assert. Report the CCCL version instead: the
   headers are compatible and only the assert breaks the build. */
#define THRUST_VERSION 200400
#define THRUST_MAJOR_VERSION     (THRUST_VERSION / 100000)
#define THRUST_MINOR_VERSION     (THRUST_VERSION / 100 % 1000)
#define THRUST_SUBMINOR_VERSION  (THRUST_VERSION % 100)
#define THRUST_PATCH_NUMBER 0
''')
lua = os.path.join(ic, 'xmake', 'hygon.lua')
s = io.open(lua, encoding='utf-8').read()
if 'THRUST_SHIM_DIR' not in s:
    s = s.replace('    local include_dirs = {',
                  '    -- THRUST_SHIM_DIR: DTK thrust/cub version skew shim\n'
                  '    local shim_dir = path.join(os.projectdir(), "xmake", "thrust_shim")\n'
                  '    if os.isdir(shim_dir) then\n'
                  '        add_includedirs(shim_dir)\n'
                  '    end\n\n    local include_dirs = {', 1)
    io.open(lua, 'w', encoding='utf-8').write(s)
print('shim ok', os.path.isfile(os.path.join(shim, 'version.h')), 'THRUST_SHIM_DIR' in io.open(lua, encoding='utf-8').read())
PYEOF

# Wheels (started above, right after prep began) must be on disk before the AOT
# side is patched and imported.
if [ -n "$WHL_PID" ]; then
  wait "$WHL_PID" || { echo "FETCH_WHEELS_FAILED"; tail -5 "$W/logs/wheels.log" 2>/dev/null | cut -c1-160; exit 1; }
  echo "wheels: $(ls "$WHEELS"/*.whl 2>/dev/null | wc -l) files downloaded"
fi
if [ -n "$WHEELS" ]; then
  "$PY" - "$W/py" "$WHEELS" <<'PYEOF'
import sys, zipfile, pathlib, glob, os
dst = pathlib.Path(sys.argv[1]); dst.mkdir(parents=True, exist_ok=True)
n = 0
for w in sorted(glob.glob(os.path.join(sys.argv[2], '*.whl'))):
    with zipfile.ZipFile(w) as z:
        z.extractall(dst)
    n += 1
print('unpacked wheels:', n)
PYEOF
fi

stage "step 2/6 python AOT patches (patch_py.py: driverless target, hip .cpp sources, EXDEV cache)"
# The offline-properties edit is only correct when no device is visible; with the
# DTK torch importable the driver query returns the real gfx936 numbers, which is
# strictly better than the fallback.  Decide it here rather than by hand.
if "$PY" -c "import torch" >/dev/null 2>&1; then
  unset NINETOOTHED_OFFLINE_DEVICE
else
  export NINETOOTHED_OFFLINE_DEVICE=1
fi
echo "NINETOOTHED_OFFLINE_DEVICE=${NINETOOTHED_OFFLINE_DEVICE:-0}"
PATCHER=$(dirname "$0")/patch_py.py
if [ ! -f "$PATCHER" ]; then echo "PATCHER_MISSING: put patch_py.py next to from_scratch.sh"; exit 1; fi
echo "patcher=$PATCHER"
"$PY" "$PATCHER" "$W/py" || { echo "PATCHER_FAILED"; exit 1; }

stage "step 3/6 nine-toothed AOT kernels (flash-attention grids 72 and 128, bf16, block 256x64)"
export PATH="$DTK/bin:$PATH"
export CUDA_HOME="$DTK/cuda/cuda"
# The AFS mount rejects rename(2) between directories (triton's cache does a
# temp-file + os.replace), so the compiler scratch space must live on local
# disk.  Only the generated kernels below need to persist on AFS.
SCRATCH=${CRB_SCRATCH:-}
if [ -z "$SCRATCH" ] || ! mkdir -p "$SCRATCH/tmp" 2>/dev/null; then
  SCRATCH=$(mktemp -d) || { echo "NO_SCRATCH"; exit 1; }
  mkdir -p "$SCRATCH/tmp"
fi
mkdir -p "$SCRATCH/tc"
echo "scratch=$SCRATCH"
df -h "$SCRATCH" | tail -1
export TMPDIR="$SCRATCH/tmp" TRITON_CACHE_DIR="$SCRATCH/tc"
export PYTHONPATH="$W/py:$W/ic/src"
# libtriton is linked against the toolchain's LLVM; without these the loader
# picks up an older system libLLVM and llvm.parse_assembly rejects the emitted
# opaque-pointer IR ("addSource: error: expected type") before segfaulting.
export LD_LIBRARY_PATH="$DTK/lib:$DTK/llvm/lib:/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"
export NINETOOTHED_TARGET="hip:$ARCH:64" NINETOOTHED_DEFAULT_CONFIG=8,2
export NINETOOTHED_OFFLINE_PROPS=1 NTOPS_JOBS=1
export NINETOOTHED_CC="$DTK/bin/hipcc"
# PYTHONFAULTHANDLER makes the AOT compile subprocess dump a python traceback
# instead of dying silently if it hits a signal.
export PYTHONFAULTHANDLER=1
"$PY" -c "import triton; print('triton', triton.__version__)" 2>&1 | tail -2
# LLVM codegen of the flash-attention kernel recurses deeply; the default 8 MB
# stack overflows (SIGSEGV) in some containers.
echo "ulimit -s = $(ulimit -s)"
ulimit -s unlimited 2>/dev/null || ulimit -s 262144 2>/dev/null
echo "ulimit -s -> $(ulimit -s)"
"$PY" -m triton.tools.compile --help >/dev/null 2>&1
echo "COMPILE_HELP_RC=$?"
export NINETOOTHED_CFLAGS="-I$CUDA_HOME/targets/x86_64-linux/include -L$CUDA_HOME/targets/x86_64-linux/lib -std=c++17"
rm -rf "$W/ic/build/ninetoothed"/*
mkdir -p "$W/ic/build/ninetoothed"
if [ -n "${CRB_KERNELS_TGZ:-}" ] && [ -f "${CRB_KERNELS_TGZ}" ]; then
  # Fallback: a pre-generated kernel archive (ic/build/ninetoothed/*), produced by
  # this same script on a node where the AOT compiler could run.  See build/README.md.
  stage "step 3b/6 pre-generated kernels ($CRB_KERNELS_TGZ)"
  tar tzf "$CRB_KERNELS_TGZ" | head -12
  tar xzf "$CRB_KERNELS_TGZ" -C "$W" || { echo "KERNELS_TGZ_FAILED"; exit 1; }
  if [ "$(ls "$W/ic/build/ninetoothed" | wc -l)" -eq 0 ]; then
    echo "KERNELS_TGZ_LAYOUT_RETRY (archive is flat, not rooted at ic/build/ninetoothed)"
    tar xzf "$CRB_KERNELS_TGZ" -C "$W/ic/build/ninetoothed" || { echo "KERNELS_TGZ_FAILED"; exit 1; }
  fi
  echo "KERNEL_FILES=$(ls "$W/ic/build/ninetoothed" | wc -l)"
  echo "KERNEL_HDR=$(ls "$W/ic/build/ninetoothed"/*.h 2>/dev/null | wc -l)"
  ls -l "$W/ic/build/ninetoothed" | head -8
  NTOPS_RC=0
else
  cd "$W/ic" || exit 1
  # Only the flash-attention kernels are needed: the other nine-toothed ops stay
  # on their default backends because their INFINIOP_*_NINETOOTHED_* macros are
  # not defined at configure time (see step 5).
  "$PY" -u -c "import infiniop.ops.flash_attention.ninetoothed.build as b; b.build()" >"$W/aot.log" 2>&1
  NTOPS_RC=$?
  echo "NTOPS_RC=$NTOPS_RC"
  tail -25 "$W/aot.log"
  echo "KERNEL_FILES=$(ls "$W/ic/build/ninetoothed" | wc -l)"
  ls -l "$W/ic/build/ninetoothed" | head -8
fi
# CRB_ONLY_AOT stops right here: the AOT step is the fragile one, and a job that
# only checks whether the kernels come out costs a fraction of a full build.
if [ "${CRB_ONLY_AOT:-0}" = 1 ]; then echo "STOP_AFTER_AOT NTOPS_RC=$NTOPS_RC"; exit 0; fi

stage "step 3c/6 dispatcher sources (flash_attention.c/.h over the two grids)"
if [ ! -f "$W/ic/build/ninetoothed/flash_attention.h" ]; then
  GD=$(dirname "$0")/gen_dispatch.py
  [ -f "$GD" ] || { echo "GEN_DISPATCH_MISSING"; exit 1; }
  ( cd "$W/ic" && PYTHONPATH="$W/py:$W/ic/src" "$PY" -u "$GD" ) || { echo "GEN_DISPATCH_FAILED"; exit 1; }
fi
ls -l "$W/ic/build/ninetoothed"/flash_attention.[ch] || { echo "DISPATCH_SOURCES_MISSING"; exit 1; }

stage "step 4/6 generated source fixes (warpSize is 64 on gfx936, hip context call)"
"$PY" - "$W" <<'PYEOF'
import glob, io, os, sys
W = sys.argv[1]
n = tot = 0
for f in glob.glob(os.path.join(W, 'ic', 'build', 'ninetoothed', '*.cpp')):
    tot += 1
    s = io.open(f, encoding='utf-8', errors='replace').read(); o = s
    s = s.replace('warpSize', '64').replace('cuCtxGetId', 'hipCtxGetId')
    if s != o:
        io.open(f, 'w', encoding='utf-8').write(s); n += 1
print('kernel sources rewritten:', n, 'of', tot)
PYEOF

else
  echo "=== PHASE=build: reusing kernels and patched sources under $W/ic"
fi

stage "step 5a/6 xmake self-patches (tolerant mkdir on AFS, build cache off)"
# 官方 bundle 的 Lua 程序树是内嵌的，没有可补丁的文件：此时 XM_SHARE 为空或不存在。
if [ -n "${XM_SHARE:-}" ] && [ -d "$XM_SHARE" ]; then
"$PY" - "$XM_SHARE" <<'PYEOF'
import io, os, sys
X = sys.argv[1]
p = os.path.join(X, 'core/base/os.lua')
if os.path.isfile(p):
    s = io.open(p, encoding='utf-8').read()
    old = '        if not os._mkdir(_dir) then'
    if 'P31_MKDIR_TOLERANT' in s:
        print('os.lua already patched')
    elif old in s:
        s = s.replace(old, '        if not os._mkdir(_dir) and not os.isdir(_dir) then', 1)
        s = s.replace('            return false, string.format("cannot create directory: %s, %s", _dir, os.strerror())',
                      '            return false, string.format("cannot create directory: %s, %s", _dir, os.strerror()) -- P31_MKDIR_TOLERANT', 1)
        io.open(p, 'w', encoding='utf-8').write(s); print('os.lua patched')
    else:
        print('os.lua PATTERN_MISS')
p = os.path.join(X, 'modules/private/cache/build_cache.lua')
if os.path.isfile(p):
    s = io.open(p, encoding='utf-8').read()
    anchor = 'function is_enabled(target)\n'
    if 'return false -- P32_CACHE_OFF' in s:
        s = s.replace('    return false -- P32_CACHE_OFF', '    do return false end -- P32_CACHE_OFF')
        io.open(p, 'w', encoding='utf-8').write(s)
        print('cache line repaired')
    elif 'P32_CACHE_OFF' in s:
        print('cache already off')
    elif anchor in s:
        io.open(p, 'w', encoding='utf-8').write(s.replace(anchor, anchor + '    do return false end -- P32_CACHE_OFF\n', 1))
        print('cache disabled')
    else:
        print('cache PATTERN_MISS')
PYEOF
else
  echo "xmake program dir absent (bundle mode) -- self-patches skipped"
fi
stage "step 5/6 xmake configure"
# xmake resolves the toolchain from PATH; without the DTK llvm directory it picks
# up the system g++ (GCC pragma diagnostics are errors there: -Werror=unknown-pragmas).
export PATH="$DTK/llvm/bin:$DTK/bin:$PATH"
echo "which clang=$(command -v clang) which hipcc=$(command -v hipcc || echo none)"
# --omp=n keeps libomp out of the link, but a few CPU operators include <omp.h>
# unconditionally.  Any compiler-shipped copy satisfies the include (nothing calls
# into the runtime because _OPENMP stays undefined).
OMP_INC=""
for c in /usr/lib/gcc/x86_64-linux-gnu/*/include /usr/lib/gcc/*/*/include /opt/dtk/llvm/lib/clang/*/include /opt/dtk/lib/clang/*/include; do
  if [ -f "$c/omp.h" ]; then OMP_INC="-I$c"; break; fi
done
echo "OMP_INC=${OMP_INC:-none}"
export XMAKE_ROOT=y XMAKE_GLOBALDIR="$W/xmake-global"
# 源码编译出的 xmake 要指向自己的 Lua 程序树；官方 bundle 自带程序树，
# 硬塞一个不存在的 XMAKE_PROGRAM_DIR 会让它启动失败。
if [ -n "${XM_SHARE:-}" ] && [ -d "$XM_SHARE" ]; then export XMAKE_PROGRAM_DIR="$XM_SHARE"; fi
export INFINI_ROOT="$W/.infini"
cd "$W/ic" || exit 1
"$XM_BIN" f -y -m release --hygon-dcu=y --hygon-arch="$ARCH" \
  --cuda="$DTK/cuda/cuda" --aten=y --omp=n --ccl=y --ninetoothed=y \
  --cxflags="-DINFINIOP_HYGON_NINETOOTHED_FLASH_ATTENTION -DTHRUST_IGNORE_CUB_VERSION_CHECK -D__HIP_PLATFORM_AMD__ -Wno-error=unknown-pragmas -Wno-error=unused-result -Wno-error=unused-variable -Wno-error=unused-but-set-variable -Wno-error=unused-function $OMP_INC" || echo "CFG_RC=$?"

stage "step 6/6 xmake build"
"$XM_BIN" -y -j "$J" || echo "BUILD_RC=$?"

# The python binding is declared set_default(false), so the plain build above skips
# it.  Stage what `xmake install` would stage (headers + the libraries just built)
# and then build that target explicitly.
stage "step 6b/6 python binding (_infinicore)"
mkdir -p "$W/.infini/include" "$W/.infini/lib"
cp -f "$W/ic/include/"*.h "$W/ic/include/"*.hpp "$W/.infini/include/" 2>/dev/null
cp -f "$W/ic/build/linux/x86_64/release/"*.so "$W/.infini/lib/" 2>/dev/null
"$XM_BIN" -y -j "$J" _infinicore || echo "BIND_RC=$?"
find "$W/ic/build" -name "_infinicore*.so" 2>/dev/null | head -3

echo "=== artifacts"
ls -l "$W"/ic/build/linux/x86_64/release/*.so 2>/dev/null | head
ls "$W/ic/build/ninetoothed" 2>/dev/null | wc -l
stage "done"
