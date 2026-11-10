#!/bin/bash
# repro_xmake_step.sh -- one step of the from-scratch xmake toolchain build.
#
# The cluster gives a compute node no writable local disk, and the AFS home is
# slow for small files (measured: copying 4008 files takes 732 s; make produces
# ~16 objects/min), while a single job is capped at 3200 s and xmake needs 860
# objects.  So this step does as much as fits, packs whatever it has into
# chain/<ts>/xmake-toolchain.tgz, and the next job continues from that archive.
# Re-run until the script prints XMP_OK.
#
# Scratch preference: a real tmpfs (TMPDIR / /dev/shm / /tmp / /var/tmp) so the
# object files never touch AFS; AFS only sees the final archive.
set -uo pipefail
H=${HOME}/minimax_h3
CH=$H/chain/$(date -u +%Y%m%d_%H%M%S)
BUDGET=${BUDGET:-2600}
mkdir -p "$CH"
T0=$(date +%s)
el() { echo $(( $(date +%s) - T0 )); }
echo "JOB_START $(date -u +%FT%T)"
echo "CH $CH"
echo "NPROC $(nproc) BUDGET $BUDGET"

# ---- 0) already finished in an earlier step? --------------------------------
PREV=$(find "$H/chain" -maxdepth 3 -name xmake-toolchain.tgz -type f 2>/dev/null | head -3)
for p in $PREV; do
  if tar tzf "$p" 2>/dev/null | grep -q 'bin/xmake$'; then
    echo "XM_STEP_DONE archive=$p bytes=$(stat -c%s "$p")"
    cp -f "$p" "$CH/xmake-toolchain.tgz" 2>/dev/null || true
    echo "XMP_OK $(date -u +%FT%T)"
    exit 0
  fi
done

# ---- 1) pick scratch (tmpfs first) ------------------------------------------
WD=""
for c in "${TMPDIR:-}" "${IFAB_TMPDIR:-}" /dev/shm/infinifabric-preparation /dev/shm /tmp /var/tmp "$H/tmp/xmwork"; do
  [ -n "$c" ] || continue
  if mkdir -p "$c/xmstep" 2>/dev/null && touch "$c/xmstep/.w" 2>/dev/null; then
    WD="$c/xmstep"; echo "WORKDIR $WD $(df -h "$c" 2>/dev/null | tail -1)"; break
  fi
  echo "SKIP $c"
done
[ -n "$WD" ] || { echo "NO_WORKDIR"; exit 1; }
case "$WD" in /dev/shm*|/tmp*|/var/tmp*) echo "LOCAL_SCRATCH=1";; *) echo "LOCAL_SCRATCH=0";; esac

# ---- 2) seed sources: best partial tree, else fresh download ----------------
SEED=""
BESTO=0
for d in $(find "$H/chain" -maxdepth 4 -type f -name configure -path '*/xmake-*/*' 2>/dev/null | sed 's#/configure$##'); do
  n=$(find "$d" -name '*.o' 2>/dev/null | wc -l)
  echo "SEED_PROBE $d objs=$n"
  if [ "$n" -gt "$BESTO" ]; then BESTO=$n; SEED=$d; fi
done
if [ -n "$SEED" ]; then
  echo "SEED_PICK $SEED objs=$BESTO"
  mkdir -p "$WD/xm"
  S0=$(date +%s)
  ( cd "$SEED" && tar cf - . ) | ( cd "$WD/xm" && tar xf - ) || { echo "SEED_COPY_FAIL"; exit 1; }
  echo "SEED_COPY_SECS $(( $(date +%s) - S0 ))"
  XSRC="$WD/xm"
else
  echo "SEED_NONE fresh download"
  TGZ="$WD/xmake-v3.1.2.tar.gz"
  TGZ="$TGZ" python3 - <<'PYEOF'
import os, sys, urllib.request
dst = os.environ["TGZ"]
u = "https://github.com/xmake-io/xmake/releases/download/v3.1.2/xmake-v3.1.2.tar.gz"
for url in (u, "https://ghproxy.net/" + u, "https://gh-proxy.com/" + u, "https://gitclone.com/github.com/xmake-io/xmake/releases/download/v3.1.2/xmake-v3.1.2.tar.gz"):
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        if len(data) < 1000000:
            raise RuntimeError("short body %d" % len(data))
        open(dst, "wb").write(data); print("XM_TGZ_BYTES", len(data), url); break
    except Exception as exc:
        print("XM_TGZ_FAIL", url, exc)
else:
    sys.exit(1)
PYEOF
  [ -s "$TGZ" ] || { echo "XM_TGZ_MISSING"; exit 1; }
  mkdir -p "$WD/xm"
  tar xzf "$TGZ" -C "$WD/xm" || { echo "XM_UNPACK_FAIL"; exit 1; }
  XSRC=$(dirname "$(find "$WD/xm" -maxdepth 4 -type f -name configure -perm -u+x | head -1)")
fi
[ -n "$XSRC" ] && [ -d "$XSRC" ] || { echo "XM_SRC_MISSING"; exit 1; }
echo "XSRC $XSRC csources=$(find "$XSRC" -name '*.c' | wc -l) objs=$(find "$XSRC" -name '*.o' | wc -l)"

# ---- 3) configure if needed -------------------------------------------------
if [ ! -f "$XSRC/Makefile" ]; then
  S0=$(date +%s)
  ( cd "$XSRC" && ./configure ) > "$CH/configure.log" 2>&1
  echo "XM_CONFIGURE_SECS $(( $(date +%s) - S0 )) rc=$?"
fi
[ -f "$XSRC/Makefile" ] || { echo "XM_MAKEFILE_MISSING"; tail -10 "$CH/configure.log" 2>/dev/null | cut -c1-160; exit 1; }

# ---- 4) make with a hard stop that still leaves a packable tree -------------
J=${J:-$(nproc)}
echo "XM_MAKE_START $(date -u +%FT%T) j=$J"
( cd "$XSRC" && make -j"$J" ) > "$CH/make.log" 2>&1 &
MPID=$!
RC=""
while kill -0 "$MPID" 2>/dev/null; do
  sleep 30
  echo "XM_MAKE $(date -u +%T) el=$(el) objs=$(find "$XSRC" -name '*.o' | wc -l) lines=$(wc -l < "$CH/make.log" 2>/dev/null)"
  if [ "$(el)" -gt "$BUDGET" ]; then
    echo "XM_BUDGET_HIT el=$(el) -- stopping make, packing partial tree"
    kill -TERM "$MPID" 2>/dev/null; sleep 5; kill -9 "$MPID" 2>/dev/null
    RC="budget"
    break
  fi
done
if [ -z "$RC" ]; then wait "$MPID"; MRC=$?; echo "XM_MAKE_END rc=$MRC el=$(el)"; fi

BIN="$XSRC/build/xmake"
if [ -x "$BIN" ]; then
  "$BIN" --version || true
  OUT="$CH/xmake-toolchain.tgz"
  mkdir -p "$WD/pack"
  cp -f "$BIN" "$WD/pack/xmake"
  # xmake 运行时要自己的 Lua 程序树，一起带上
  if [ -d "$XSRC/xmake" ]; then ( cd "$XSRC" && tar cf "$WD/pack/xmake-prog.tar" xmake ); fi
  ( cd "$WD/pack" && tar cf - . ) > "$OUT.raw" && mv -f "$OUT.raw" "$OUT"
  echo "XMP_ARCHIVE $OUT bytes=$(stat -c%s "$OUT")"
  sha256sum "$OUT" | tee "$CH/MANIFEST.txt"
  chmod -R a+rwX "$CH" 2>/dev/null
  echo "XMP_OK $(date -u +%FT%T) el=$(el)"
  exit 0
fi
echo "XMP_PARTIAL el=$(el) objs=$(find "$XSRC" -name '*.o' | wc -l) -- rerun this script to continue"
tail -6 "$CH/make.log" 2>/dev/null | cut -c1-160
chmod -R a+rwX "$CH" 2>/dev/null
exit 3
