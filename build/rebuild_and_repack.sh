#!/bin/bash
set +u
echo "=== mj31 start $(date -u +%FT%T)"
PROJ=$PWD
AFS=${IFAB_AFS_HOME:-$HOME}
H=$AFS/minimax_h3
B=$H/b1
export TMPDIR=$B/tmp; mkdir -p "$TMPDIR" "$B/logs" "$H/output2" "$H/artifacts"
export XMAKE_PROGRAM_DIR=$B/xm2/xmp/share/xmake
export XMAKE_GLOBALDIR=$B/xg2
export XMAKE_ROOT=y
export INFINI_ROOT=$B/.infini2
export PYBIND11_ROOT=$H/tools/pybind11-3.1.0
export PYTHON_INCLUDE=/usr/local/include/python3.11
export PYTHON_LIBDIR=/usr/local/lib
export PYTHON_LIBNAME=python3.11
export CUDA_HOME=/opt/dtk/cuda/cuda
export PATH=$B/xm2/xmp/bin:$PATH
export LD_LIBRARY_PATH=/opt/dtk/lib:/opt/dtk/llvm/lib:/opt/hyhal/lib:/opt/dtk/.hyhal/rocm_smi:/opt/dtk/.hyhal/rocm_smi/lib:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
R=$B/ic/build/linux/x86_64/release
echo "--- step1 patch bridge"
python3 -u "$PROJ/p40_pad80.py" "$B" 2>&1 | tail -6
echo "--- step2 incremental build"
XM=$(command -v xmake); echo "XM=$XM"
cd "$B/ic" || exit 13
date -u +%FT%T
timeout 1000 $XM -y -j 24 > "$B/logs/bld31.log" 2>&1
echo "BLD_RC=$?"
echo "ERRCNT=$(grep -cE '(^|: )error:' "$B/logs/bld31.log")"
grep -E "(^|: )error:|fatal error" "$B/logs/bld31.log" | head -20 | cut -c1-190
tail -3 "$B/logs/bld31.log" | cut -c1-160
ls -l $R/libinfinicore_cpp_api.so $R/libinfiniop.so 2>/dev/null | awk '{print $5, $9}'
echo "--- step3 repack artifacts"
if [ -f $R/libinfinicore_cpp_api.so ]; then
  cp -f "$H/artifacts/infinicore.tgz" "$H/artifacts/infinicore_prev.tgz" 2>/dev/null
  python3 -u "$PROJ/repack_ic.py" "$H/artifacts/infinicore_prev.tgz" "$R" "$H/artifacts/infinicore.tgz" 2>&1 | tail -3
fi
echo "--- step4 fa80 numeric check"
LIB=$B/ic/python/infinicore/lib
mkdir -p $LIB
for f in _infinicore.cpython-311-x86_64-linux-gnu.so libinfiniop.so libinfinicore_cpp_api.so libinfinirt.so libinfiniccl.so; do cp -f $R/$f $LIB/ 2>/dev/null; done
echo "LIB_COUNT=$(ls $LIB | wc -l)"
export PYTHONPATH=$B/ic/python
export LD_LIBRARY_PATH=$LIB:$B/.infini2/lib:/opt/dtk/lib:/opt/dtk/llvm/lib:/opt/hyhal/lib:/opt/dtk/.hyhal/rocm_smi:/opt/dtk/.hyhal/rocm_smi/lib
cp -f $PROJ/fa80.py $B/ 2>/dev/null
cd $B
timeout 700 python3 -u fa80.py > $B/logs/fa80_31.log 2>&1
echo "FA80_RC=$?"
grep -E "^CASE" $B/logs/fa80_31.log | cut -c1-150
grep -E "FA80_DONE" $B/logs/fa80_31.log | tail -1
tail -5 $B/logs/fa80_31.log | cut -c1-170
cp -f $B/logs/fa80_31.log $H/output2/fa80_31.log 2>/dev/null
echo "=== mj31 done $(date -u +%FT%T)"
