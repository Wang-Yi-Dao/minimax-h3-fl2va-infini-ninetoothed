#!/bin/bash
# crb_pub: reproducibility check that follows the PUBLISHED instructions only.
# It uses exactly the three artefacts the repository/release ships --
#   src/InfiniCore-snapshot-mit.tar.gz   (MIT upstream snapshot)
#   src/overlay.tar.gz                   (the nine changed sources)
#   src/kernels_deployed.tgz             (the nine-toothed kernel set)
# verifies each sha256 against the published value, rebuilds the five shared
# libraries from them in a fresh local scratch tree, then runs the published
# 50-step FL2VA pipeline and the official compare with those libraries.
set -uo pipefail
PROJ=$PWD
AFS=${IFAB_AFS_HOME:-$HOME}
H=$AFS/minimax_h3
SH=/public/home/aclsylqidf/shared/MiniMax-H3/diffusers
BW=$PROJ/.build
RW=$PROJ/.run
echo "JOB_START $(date -u +%FT%T) host=$(hostname) crb-pub"
df -h "$PROJ" | tail -1
echo "PUB_SNAP_SHA=620d40e299ed25d154903d09242b5b1dc5ec37f7c971a7ed116902217a029fdd"
echo "PUB_OVERLAY_SHA=e66b78c22385b4ae5d6c80c3ce28c9db72c1ff5159f0c26f5094fd781e5ce754"
echo "PUB_KERNELS_SHA=fbf982472e7f195038c6f4225448d6b7c82a358f2092c363278c685423e821cc"
echo "=== A. stage the published artefacts and verify them"
cp -f "$H/src/InfiniCore.tgz" "$PROJ/snapshot.tgz" || { echo SNAP_COPY_FAIL; exit 0; }
rm -rf "$PROJ/kernels"; mkdir -p "$PROJ/kernels"
cp -f "$H/crb/kernels_deployed.tgz" "$PROJ/kernels.tgz" || { echo KERNELS_COPY_FAIL; exit 0; }
cat $(ls "$PROJ"/ov.p* | sort -V) | base64 -d > "$PROJ/overlay.tar.gz" || { echo B64_FAIL; exit 0; }
for pair in "snapshot.tgz:620d40e299ed25d154903d09242b5b1dc5ec37f7c971a7ed116902217a029fdd" \
            "overlay.tar.gz:e66b78c22385b4ae5d6c80c3ce28c9db72c1ff5159f0c26f5094fd781e5ce754" \
            "kernels.tgz:fbf982472e7f195038c6f4225448d6b7c82a358f2092c363278c685423e821cc"; do
  f=${pair%%:*}; want=${pair##*:}
  got=$(sha256sum "$PROJ/$f" | cut -d" " -f1)
  if [ "$got" = "$want" ]; then echo "SHA_OK $f"; else echo "SHA_MISMATCH $f got=$got"; exit 0; fi
done
tar tzf "$PROJ/kernels.tgz" | head -3
echo "KERNEL_ENTRIES=$(tar tzf "$PROJ/kernels.tgz" | grep -c '\.cpp$')"
echo "=== B. rebuild with from_scratch.sh (published script, published inputs)"
mkdir -p "$BW" "$RW" "$PROJ/base" "$PROJ/.scratch/tmp"
chmod 0644 "$PROJ/build.sh" "$PROJ/patch_py.py" "$PROJ/gen_dispatch.py" 2>/dev/null
export W=$BW DTK=/opt/dtk
export IC_TGZ=$PROJ/snapshot.tgz OVL_TGZ=$PROJ/overlay.tar.gz XMAKE_TGZ=$H/xmake312.tgz
export CRB_KERNELS_TGZ=$PROJ/kernels.tgz WHEELS=$HOME/whl PYBIND11_ROOT=$H/tools/pybind11-3.1.0
export J=32 ARCH=gfx936 PHASE=all CRB_RESUME=0 CRB_SCRATCH=$PROJ/.scratch
LOG=$PROJ/build.log
( timeout 1500 bash "$PROJ/build.sh" ) > "$LOG" 2>&1 &
PY=$!; LASTR=0
while kill -0 $PY 2>/dev/null; do
  sleep 15; NOW=$(date +%s)
  if [ $((NOW-LASTR)) -ge 90 ]; then LASTR=$NOW
    echo "--- $(date -u +%T) build $(wc -l < "$LOG") lines $(grep -c 'error:' "$LOG") errors: $(grep -aE '^=== \[' "$LOG" | tail -1)"; fi
done
wait $PY; echo "BUILD_EXIT=$?"
grep -aE "SHA_OK|KERNEL_FILES|KERNEL_HDR|DISPATCH|BIND_RC|PATCH .* (OK|MISS)|error:" "$LOG" | tail -12 | cut -c1-180
FR=$BW/ic/build/linux/x86_64/release
echo "FRESH_SO_COUNT=$(ls "$FR"/*.so 2>/dev/null | wc -l)"
ls -l "$FR"/*.so 2>/dev/null | awk '{print "FRESH_SO", $5, $9}'
if [ ! -f "$FR/libinfiniop.so" ] || [ ! -f "$FR/_infinicore.cpython-311-x86_64-linux-gnu.so" ]; then
  echo BUILD_INCOMPLETE; grep -a "error:" "$LOG" | head -4 | cut -c1-180; echo "JOB_END $(date -u +%FT%T)"; exit 0
fi
echo "=== C. published run: 50 steps + official compare with the rebuilt libraries"
cd "$PROJ" || exit 1
IFAB_WORKSPACE_ROOT=$RW CRB_LIBS=$FR B=$PROJ/base timeout 1700 bash "$PROJ/nn38.sh" 50 1600 2>&1 | tail -34
echo "JOB_END $(date -u +%FT%T)"
