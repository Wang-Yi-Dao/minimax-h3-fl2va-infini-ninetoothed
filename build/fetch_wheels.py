#!/usr/bin/env python3
"""fetch_wheels.py -- materialise the four wheels the AOT step imports.

    usage: fetch_wheels.py <dest dir> [local triton wheel]

Everything is unzipped into <dest dir>, so a single call yields the import tree
that patch_py.py then edits.  The file URLs come from PyPI's JSON metadata:
the platform image ships pip 26.0.1, whose resolver dies with an internal
traceback on these downloads, and ninetoothed/numpy need different platform
selectors anyway (ninetoothed 0.26.0 is py3-none-any, so asking for it with a
manylinux --platform matches no file at all).

triton is the Hygon build: pypi.org's triton has no hip:gfx936 backend, so the
AOT step could not emit a kernel.  Use the plain
3.5.1+das.opt1.dtk2604.torch290 wheel, NOT the date-stamped rebuilds -- those
pass -mllvm flags that the container's only clang (17) rejects, and clang then
exits 0 without writing the .amdgcn file.
"""
import json
import os
import pathlib
import shutil
import sys
import urllib.request
import zipfile

DST = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "wheels")
LOCAL_TRITON = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None
DST.mkdir(parents=True, exist_ok=True)

TRITON_URL = os.environ.get(
    "TRITON_WHEEL_URL",
    "https://pypi.sourcefind.cn/packages/das_release/dtk2604-rc1/e39/7b16365da2250/"
    "triton-3.5.1%2Bdas.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl",
)


def _linux_cp311(name):
    # A wheel for this platform only: manylinux, x86_64, cp311 -- never the
    # macosx / aarch64 builds that also carry "cp311" in their file name.
    return (name.endswith(".whl") and "cp311" in name and "manylinux" in name
            and "x86_64" in name and "macosx" not in name)


# (package, version or None for latest, filename selector)
WANT = [
    ("ninetoothed", "0.26.0", lambda n: n.endswith(".whl")),
    ("sympy", None, lambda n: n.endswith(".whl")),
    ("mpmath", None, lambda n: n.endswith(".whl")),
    ("numpy", "1.26.4", _linux_cp311),
]


def get(url, timeout=120, tries=3):
    last = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as exc:                        # retry a flaky mirror
            last = exc
    raise last


def unzip(path):
    with zipfile.ZipFile(path) as z:
        z.extractall(DST)
    return len(z.namelist())


def main():
    for pkg, ver, ok in WANT:
        base = ("https://pypi.org/pypi/%s/json" % pkg) if ver is None else \
               ("https://pypi.org/pypi/%s/%s/json" % (pkg, ver))
        try:
            meta = json.loads(get(base))
        except Exception as exc:
            print("META_FAIL %s %s" % (pkg, exc))
            continue
        files = [f for f in meta["urls"] if ok(f["filename"])]
        if not files:
            print("NO_FILE %s %s" % (pkg, ver))
            continue
        f = files[0]
        try:
            blob = get(f["url"], timeout=300)
        except Exception as exc:
            print("DL_FAIL %s %s" % (f["filename"], exc))
            continue
        out = DST / f["filename"]
        out.write_bytes(blob)
        n = unzip(out)
        print("OK %-64s %9d B  %3d entries" % (f["filename"], len(blob), n))

    triton = LOCAL_TRITON if (LOCAL_TRITON and LOCAL_TRITON.is_file()) else None
    if triton is None:
        triton = DST / "triton351.whl"
        if not triton.is_file() or triton.stat().st_size < 1000:
            try:
                triton.write_bytes(get(TRITON_URL, timeout=600))
            except Exception as exc:
                print("META_FAIL triton %s" % exc)
                return 1
    n = unzip(triton)
    print("OK %-64s %9d B  %3d entries" % ("triton351.whl (Hygon 3.5.1)",
                                           triton.stat().st_size, n))
    print("TREE ninetoothed=%s triton=%s numpy=%s" % (
        (DST / "ninetoothed").is_dir(), (DST / "triton").is_dir(),
        (DST / "numpy").is_dir()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
