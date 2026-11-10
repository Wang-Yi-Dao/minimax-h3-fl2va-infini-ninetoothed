#!/bin/bash
# xmake_build.sh -- compile xmake 3.1.2 from its release source tree once, then
# pack the whole in-place build into a single tgz that later builds can unpack.
#
# Why this is a separate step: compiling xmake writes thousands of object files,
# which costs ~30 min on the AFS-backed persistent storage this cluster gives a
# compute node (the node offers no writable local disk at all).  A single job is
# capped at 3200 s, so the toolchain build gets its own job and hands the result
# forward as one read-only archive.
#
#   W        work directory (xmake source is unpacked under $W/xm)
#   XM_TGZ   xmake-v3.1.2.tar.gz (release source tree)
#   OUT      output archive (default $W/deps/xmake-built.tgz)
#   J        make parallelism (default: nproc)
set -uo pipefail
W=${W:?set W}
XM_TGZ=${XM_TGZ:?set XM_TGZ}
OUT=${OUT:-$W/deps/xmake-built.tgz}
J=${J:-$(nproc)}
mkdir -p "$W/xm" "$(dirname "$OUT")"
XM_BIN=$(find "$W/xm" -maxdepth 5 -type f -name xmake -perm -u+x 2>/dev/null | head -1)
if [ -z "$XM_BIN" ] && [ -s "$OUT" ]; then
  echo "XM_REUSE $OUT $(stat -c%s "$OUT")"; exit 0
fi
# 官方 bundle 优先：xmake v3.1.2 的 release 资产里有
# xmake-bundle-v3.1.2.linux.x86_64（2.8 MB，自带 Lua 程序树，解压即用）。
# 它若能在本镜像上跑起来，源码编译 xmake 那一段（AFS 上约 30 分钟）整段省掉；
# 跑不起来（例如 glibc 太旧）再回退源码编译。两条路写在同一作业里，
# 这样一次尝试就能确定用哪条，不必单独投探测作业。
PYB=${PY:-python3}
XM_BUNDLE=${XM_BUNDLE-$W/deps/xmake-bundle-v3.1.2.linux.x86_64}
# 官方 bundle 是 glibc 2.38 链接的（实测 ELF 里引用 GLIBC_2.38）；
# 本镜像的 glibc 更旧，直接跳过下载，省 2.8 MB 和一次失败等待。
GLIBCV=$(ldd --version 2>/dev/null | head -1 | grep -oE "[0-9]+\.[0-9]+$" | head -1)
if [ -n "$GLIBCV" ] && [ "$(echo "$GLIBCV 2.38" | awk '{print ($1<$2)?0:1}')" = 0 ]; then
  echo "XM_BUNDLE_SKIP glibc=$GLIBCV < 2.38"
  XM_BUNDLE=/dev/null
else
  echo "XM_BUNDLE_TRY glibc=${GLIBCV:-unknown}"
fi
if [ -z "$XM_BIN" ] && [ ! -s "$XM_BUNDLE" ]; then
  XM_BUNDLE="$XM_BUNDLE" "$PYB" - <<'PYEOF' || true
import os, sys, urllib.request
dst = os.environ["XM_BUNDLE"]
url = "https://github.com/xmake-io/xmake/releases/download/v3.1.2/xmake-bundle-v3.1.2.linux.x86_64"
for u in (url, "https://ghproxy.net/" + url, "https://gh-proxy.com/" + url):
    try:
        with urllib.request.urlopen(u, timeout=150) as r:
            data = r.read()
        if len(data) < 1000000:
            raise RuntimeError("short body %d" % len(data))
        os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
        open(dst, "wb").write(data)
        print("XM_BUNDLE_BYTES", len(data), u)
        break
    except Exception as exc:
        print("XM_BUNDLE_DL_FAIL", u, exc)
else:
    sys.exit(1)
PYEOF
fi
if [ -z "$XM_BIN" ] && [ -s "$XM_BUNDLE" ]; then
  chmod +x "$XM_BUNDLE" 2>/dev/null
  BOUT=$(timeout 90 "$XM_BUNDLE" --version 2>&1 | head -3); BRC=$?
  echo "XM_BUNDLE_TEST rc=$BRC ver=$(echo "$BOUT" | tr '\n' ' ' | cut -c1-90)"
  if [ "$BRC" = 0 ] && echo "$BOUT" | grep -q "3\.1\.2"; then
    mkdir -p "$W/xm/bin"
    cp -f "$XM_BUNDLE" "$W/xm/bin/xmake"; chmod +x "$W/xm/bin/xmake"
    XM_BIN="$W/xm/bin/xmake"; echo "XM_MODE=bundle"
  else
    echo "XM_MODE=source (bundle unusable)"
  fi
fi
if [ -z "$XM_BIN" ]; then
  tar xzf "$XM_TGZ" -C "$W/xm" || { echo XM_UNPACK_FAIL; exit 1; }
  XSRC=$(find "$W/xm" -maxdepth 4 -type f -name configure -perm -u+x 2>/dev/null | head -1)
  [ -n "$XSRC" ] || { echo XM_CONFIGURE_MISSING under "$W/xm"; exit 1; }
  XSRC=$(dirname "$XSRC")
  echo "XM_CONFIGURE_START $(date -u +%FT%T) src=$XSRC"
  ( cd "$XSRC" && ./configure ) > "$W/xmake-configure.log" 2>&1 || {
    echo XM_CONFIGURE_FAIL; tail -15 "$W/xmake-configure.log" | cut -c1-160; exit 1; }
  echo "XM_MAKE_START $(date -u +%FT%T) j=$J"
  # 输出写到文件、由外层循环报进度：作业被时限杀掉时日志仍在持久盘上，
  # 可以直接看出 make 走到了哪里（直接接管道会整段缓冲，什么都看不到）。
  ( cd "$XSRC" && make -j"$J" ) > "$W/xmake-make.log" 2>&1 &
  MPID=$!
  while kill -0 "$MPID" 2>/dev/null; do
    sleep 60
    echo "XM_MAKE $(date -u +%T) lines=$(wc -l < "$W/xmake-make.log" 2>/dev/null) err=$(grep -c 'Error ' "$W/xmake-make.log" 2>/dev/null)"
  done
  wait "$MPID" || { echo XM_MAKE_FAIL; tail -20 "$W/xmake-make.log" | cut -c1-160; exit 1; }
  XM_BIN="$XSRC/build/xmake"
fi
[ -x "$XM_BIN" ] || { echo XM_NOT_BUILT; exit 1; }
echo "XM_BIN $XM_BIN"
tar czf "$OUT.tmp" -C "$W/xm" . || { echo XM_PACK_FAIL; exit 1; }
mv -f "$OUT.tmp" "$OUT"
echo "XM_PREBUILT $OUT"
echo "XM_PREBUILT_BYTES $(stat -c%s "$OUT")"
sha256sum "$OUT"
