#!/bin/bash
# repro_xmake_tmpfs.sh -- compile xmake 3.1.2 from release source inside the
# largest writable scratch this node offers (tmpfs if any, else AFS), then pack
# the finished tree into chain/<ts>/xmake-built.tgz for the InfiniCore job.
set -uo pipefail
H=${HOME}/minimax_h3
CH=$H/chain/$(date -u +%Y%m%d_%H%M%S)
mkdir -p "$CH"
echo "JOB_START $(date -u +%FT%T)"
echo "CH $CH"
echo "NPROC $(nproc)"

WD=""
for c in "${TMPDIR:-}" "${IFAB_TMPDIR:-}" /dev/shm/infinifabric-preparation /dev/shm /tmp /var/tmp "$H/tmp/xmwork"; do
  [ -n "$c" ] || continue
  if mkdir -p "$c/xmbuild" 2>/dev/null && touch "$c/xmbuild/.w" 2>/dev/null; then
    WD="$c/xmbuild"; echo "WORKDIR $WD $(df -h "$c" 2>/dev/null | tail -1)"; break
  fi
  echo "SKIP $c not writable"
done
[ -n "$WD" ] || { echo "NO_WORKDIR"; exit 1; }
case "$WD" in /dev/shm*|/tmp*|/var/tmp*) echo "LOCAL_SCRATCH=1";; *) echo "LOCAL_SCRATCH=0";; esac

TGZ="$WD/xmake-v3.1.2.tar.gz"
T0=$(date +%s)
TGZ="$TGZ" python3 - <<'PYEOF'
import os, sys, urllib.request
dst = os.environ["TGZ"]
u = "https://github.com/xmake-io/xmake/releases/download/v3.1.2/xmake-v3.1.2.tar.gz"
for url in (u, "https://ghproxy.net/" + u, "https://gh-proxy.com/" + u):
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        if len(data) < 1000000:
            raise RuntimeError("short body %d" % len(data))
        open(dst, "wb").write(data)
        print("XM_TGZ_BYTES", len(data), url)
        break
    except Exception as exc:
        print("XM_TGZ_FAIL", url, exc)
else:
    sys.exit(1)
PYEOF
[ -s "$TGZ" ] || { echo "XM_TGZ_MISSING"; exit 1; }
echo "XM_DL_SECS $(( $(date +%s) - T0 ))"

mkdir -p "$WD/xm"
T0=$(date +%s)
tar xzf "$TGZ" -C "$WD/xm" || { echo "XM_UNPACK_FAIL"; exit 1; }
XSRC=$(dirname "$(find "$WD/xm" -maxdepth 4 -type f -name configure -perm -u+x | head -1)")
[ -n "$XSRC" ] || { echo "XM_CONFIGURE_MISSING"; exit 1; }
echo "XM_UNPACK_SECS $(( $(date +%s) - T0 )) src=$XSRC"
T0=$(date +%s)
( cd "$XSRC" && ./configure ) > "$CH/xmake-configure.log" 2>&1
RC=$?
echo "XM_CONFIGURE_SECS $(( $(date +%s) - T0 )) rc=$RC"
[ "$RC" = 0 ] || { echo "XM_CONFIGURE_FAIL"; tail -10 "$CH/xmake-configure.log" | cut -c1-160; exit 1; }
J=${J:-$(nproc)}
T0=$(date +%s)
echo "XM_MAKE_START $(date -u +%FT%T) j=$J"
( cd "$XSRC" && make -j"$J" ) > "$CH/xmake-make.log" 2>&1 &
MPID=$!
while kill -0 "$MPID" 2>/dev/null; do
  sleep 45
  echo "XM_MAKE $(date -u +%T) secs=$(( $(date +%s) - T0 )) lines=$(wc -l < "$CH/xmake-make.log" 2>/dev/null) err=$(grep -c 'Error ' "$CH/xmake-make.log" 2>/dev/null || true)"
done
wait "$MPID"; MRC=$?
echo "XM_MAKE_SECS $(( $(date +%s) - T0 )) rc=$MRC"
[ "$MRC" = 0 ] || { echo "XM_MAKE_FAIL"; tail -20 "$CH/xmake-make.log" | cut -c1-160; exit 1; }
[ -x "$XSRC/build/xmake" ] || { echo "XM_BIN_MISSING"; ls -l "$XSRC/build" 2>/dev/null | head; exit 1; }
"$XSRC/build/xmake" --version || true

OUT="$CH/xmake-built.tgz"
tar czf "$OUT.tmp" -C "$WD/xm" . && mv -f "$OUT.tmp" "$OUT" || { echo "XM_PACK_FAIL"; exit 1; }
echo "XM_PREBUILT $OUT"
echo "XM_PREBUILT_BYTES $(stat -c%s "$OUT")"
sha256sum "$OUT" | tee "$CH/MANIFEST.txt"
echo "WHERE $WD" >> "$CH/MANIFEST.txt"
chmod -R a+rwX "$CH" 2>/dev/null
echo "XMP_OK $(date -u +%FT%T)"
