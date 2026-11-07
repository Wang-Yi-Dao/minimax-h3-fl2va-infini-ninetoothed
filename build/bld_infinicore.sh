#!/bin/bash
# build InfiniCore for hygon with aten + ninetoothed flash_attention
set -uo pipefail
cd "$1" || exit 90
R=$PWD
H=$HOME/minimax_h3
export XMAKE_PROGRAM_DIR=$R/xm/xmp/share/xmake
export XMAKE_GLOBALDIR=$R/xmake-global
export XMAKE_ROOT=y
export PATH=$R/xm/xmp/bin:/opt/dtk/llvm/bin:$PATH
export INFINI_ROOT=$R/.infini
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
for d in /opt/dtk/lib /opt/dtk/llvm/lib /opt/hyhal/lib /opt/dtk/.hyhal/rocm_smi /opt/dtk/.hyhal/rocm_smi/lib /opt/dtk/cuda/cuda/targets/x86_64-linux/lib; do
  [ -d "$d" ] && export LD_LIBRARY_PATH=$d:$LD_LIBRARY_PATH
done
export PYBIND11_ROOT=$H/tools/pybind11-3.1.0
export PYTHON_INCLUDE=/usr/local/include/python3.11
export PYTHON_LIBDIR=/usr/local/lib
export PYTHON_LIBNAME=python3.11
cd $R/ic/InfiniCore || exit 91
python3 $H/p14_xmake.py xmake.lua
xmake f -y -m release --hygon-dcu=y --hygon-arch=gfx936 --cuda=/opt/dtk/cuda/cuda --aten=y --omp=n --ccl=y --ninetoothed=y --cxflags="-DINFINIOP_HYGON_NINETOOTHED_FLASH_ATTENTION" 2>&1 | tail -40
echo "CONFIG_RC=$?"
