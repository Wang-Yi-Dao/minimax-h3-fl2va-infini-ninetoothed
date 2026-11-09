# Rebuilding the InfiniCore / nine-toothed side from source

This directory lets a third party rebuild, from the pinned upstream snapshot plus
the overlay files in this package, the exact `InfiniCore` shared libraries that
produced the reported run (`Average SSIM 80.012%`).  Nothing in it is a binary
blob: `src/InfiniCore-snapshot-mit.tar.gz` is the unmodified MIT upstream tree,
`src/overlay.tar.gz` holds the nine files this project changed,
`src/kernels_deployed.tgz` holds the nine-toothed kernel sources (generated C++,
not object code), and the rest is the scripts that drive the build.

## Files

| file | what it is |
|---|---|
| `from_scratch.sh` | the whole build, six named steps, environment driven |
| `patch_py.py` | the seven edits that make the stock triton / ninetoothed wheels usable offline; prints `PATCH <name> OK\|SKIP\|MISS` for each |
| `gen_dispatch.py` | emits `ic/build/ninetoothed/flash_attention.{c,h}`, the dispatcher that chains the two kernels behind one C entry point |
| `rebuild_and_validate.sh` | the one-job wrapper used for the reproducibility check: verify shas, rebuild, then run 50 steps + official compare |
| `../src/InfiniCore-snapshot-mit.tar.gz` | upstream InfiniCore snapshot (MIT, includes `LICENSE`) |
| `../src/overlay.tar.gz` | the changed / added sources, rooted at `ic/` |
| `../src/kernels_deployed.tgz` | the nine-toothed kernel sources the released library was compiled from (`ic/build/ninetoothed/`, 106 files, 2,457,374 B) |
| `../jobs/run_tp8_fl2va.sh` | the run that consumes the built libraries (50 steps + official compare) |

Every artefact above carries a published sha256 in `../src/SHA256SUMS`; the validation job
verifies all three before it compiles anything.

The overlay touches only what the nine-toothed flash-attention path needs:

```
xmake.lua
xmake/hygon.lua
src/infiniop/ninetoothed/build.py
src/infiniop/ops/flash_attention/ninetoothed/{build.py,flash_attention.py,operator.cc}
src/infiniop/ops/{swiglu/operator.cc,relu/nvidia/relu_nvidia.cu}
src/infinicore/ops/multi_head_attention/hygon_nt/mha_ninetoothed_hygon.cc   (new file)
```

## Prerequisites

* DTK 26.04 (`/opt/dtk`), `hipcc` at `/opt/dtk/bin/hipcc`, target `gfx936`
* python 3.11 with the three wheels used for the AOT step
  (`ninetoothed 0.26.0`, `triton 3.5.1+das.opt1.dtk2604.torch290`, `numpy 1.26.4`)
* xmake 3.1.2 (linux x86_64 tarball)
* pybind11 3.1.0 headers, python 3.11 headers and library
* ~8 GB of scratch space; the AOT step wants a *local* scratch directory (see below)

## Getting the build dependencies

None of the five items below is produced by this project — they are public or
vendor artefacts, pinned to the exact versions the released library was built
with.  Run this once; afterwards `WHEELS`, `XMAKE_TGZ` and `PYBIND11_ROOT`
point at what it produced.

```bash
# 1) the three wheels the AOT step imports (python 3.11, x86_64)
mkdir -p wheels
pip download ninetoothed==0.26.0 numpy==1.26.4 \
  --only-binary=:all: --python-version 311 --platform manylinux2014_x86_64 \
  -d wheels -i https://pypi.org/simple

#    triton is NOT on pypi.org.  The build needs the Hygon-patched wheel, whose
#    local version tag (+das.opt1.dtk2604.torch290) exists only on the DTK mirror:
#    stock triton has no hip:gfx936 backend, so step 3 could not emit a kernel.
pip download "triton==3.5.1+das.opt1.dtk2604.torch290" \
  --only-binary=:all: --python-version 311 --platform manylinux_2_28_x86_64 \
  -d wheels -i https://pypi.sourcefind.cn/release/dtk/simple/ \
  --trusted-host pypi.sourcefind.cn

# 2) xmake 3.1.2 (GitHub release asset; -L is required, the URL redirects)
curl -L -o xmake-3.1.2.tar.gz \
  https://github.com/xmake-io/xmake/releases/download/v3.1.2/xmake-v3.1.2.tar.gz

# 3) pybind11 3.1.0 headers — the wheel is pure headers, unzipping *is* installing
pip download pybind11==3.1.0 --no-deps -d /tmp/pb -i https://pypi.org/simple
mkdir -p /opt/pybind11-3.1.0 && unzip -q /tmp/pb/pybind11-3.1.0-*.whl -d /opt/pybind11-3.1.0
```

then

```bash
export WHEELS=$PWD/wheels
export XMAKE_TGZ=$PWD/xmake-3.1.2.tar.gz
export PYBIND11_ROOT=/opt/pybind11-3.1.0
```

Two practical notes:

* `pip download` resolves the wheel tag (cp311, manylinux_2_17 vs _2_28, x86_64)
  for you.  Hand-written URLs fail most often on exactly that tag, so prefer it
  over guessing an asset name.
* DTK 26.04 (`/opt/dtk`, `hipcc`, torch 2.9.0+das.opt1.dtk2604) is **not**
  downloadable: it is the DCU software stack baked into the platform image.  The
  build needs it in place; everything else above is fetched.

## Build

```bash
export W=$PWD/build-work
export DTK=/opt/dtk
export IC_TGZ=$PWD/../src/InfiniCore-snapshot-mit.tar.gz
export OVL_TGZ=$PWD/../src/overlay.tar.gz
export XMAKE_TGZ=/path/to/xmake-3.1.2.tar.gz
export WHEELS=/path/to/wheels            # directory holding the three .whl files
export PYBIND11_ROOT=/opt/pybind11-3.1.0
export PYTHON_INCLUDE=/usr/local/include/python3.11
export PYTHON_LIBDIR=/usr/local/lib
export PYTHON_LIBNAME=python3.11
export J=32 ARCH=gfx936
export CRB_KERNELS_TGZ=$PWD/../src/kernels_deployed.tgz   # the released kernel set
bash from_scratch.sh
```

Six steps, each logged with its own banner:

1. **thrust version shim** — DTK ships thrust 2.3.2 next to CCCL/CUB 2.4.0; upstream
   thrust turns that skew into a hard `static_assert`.  A one-header shim under
   `xmake/thrust_shim/` reports the CCCL version and is added to the include path.
2. **python AOT patches** — `patch_py.py`.  The wheels assume a live device, a
   `.c` temp file and a working `rename(2)`; a batch job that only compiles
   kernels has none of the three.  Each edit carries a marker comment, so
   `grep P29_TARGET/DEFAULT_CONFIG/HIPCPP_COMPAT/SKIP_LIB/DRIVERLESS/EXDEV_FALLBACK`
   shows what changed and why.
3. **nine-toothed kernels** — the two flash-attention kernels the released library
   was compiled from, bf16, `block_size_m=256`, `block_size_n=64`, `head_dim` 72
   and 128 (`examples/minimax_h3` uses `head_dim=80`, which is zero-padded to 128
   by `h3_pad80`).  Point `CRB_KERNELS_TGZ` at `../src/kernels_deployed.tgz` and
   step 3 unpacks them; with `CRB_KERNELS_TGZ` unset it regenerates them from the
   overlay with the AOT compiler instead (see *If the AOT step cannot run*).
   Either way these are the only nine-toothed ops in the build: the others keep
   their default backends because their `INFINIOP_*_NINETOOTHED_*` macros are not
   defined at configure time.
3c. **dispatcher sources** — `gen_dispatch.py` writes `flash_attention.c/.h`
   (`if <grid condition> return launch_<kernel>(...);` for emb_dim 72 and 128).
4. **generated-source fixes** — `warpSize` → `64` (gfx936 waves are 64 wide) and
   `cuCtxGetId` → `hipCtxGetId` in the emitted sources.
5. **xmake self patch** — tolerate a missing cache directory, disable the build
   cache, configure with
   `--hygon-dcu=y --hygon-arch=gfx936 --aten=y --omp=n --ccl=y --ninetoothed=y`
   and `-DINFINIOP_HYGON_NINETOOTHED_FLASH_ATTENTION -DTHRUST_IGNORE_CUB_VERSION_CHECK`.
6. **compile** — `xmake -y -j32`.

### If the AOT step cannot run

Step 3 shells out to `python3 -m triton.tools.compile --target hip:gfx936:64`.  On
some container images that subprocess dies with `SIGSEGV` inside
`triton/backends/amd/compiler.py:make_amdgcn`, after LLVM printed

```
addSource: error: expected type
  %234 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %233
```

i.e. the `libtriton` shipped with the wheel refuses its own opaque-pointer IR.  In
that case point `CRB_KERNELS_TGZ` at a pre-generated kernel archive (any
`ic/build/ninetoothed/*` tree produced by this same script on a node where the AOT
compiler worked) and step 3 becomes a binary-free fallback:

```bash
export CRB_KERNELS_TGZ=/path/to/kernels.tgz
bash from_scratch.sh
```

The kernels are pure output of `src/infiniop/ops/flash_attention/ninetoothed/flash_attention.py`
from the overlay — the grid, the block sizes and the dtype are all in that file — so
the archive carries no information that is not already in this package.

Set `PHASE=build` to skip steps 1–4 on a re-run, and `CRB_RESUME=1` to reuse an
already extracted snapshot (the tree is ~7.5k files; re-extracting it dominates
the wall clock on a networked home directory).  The overlay is re-applied on
every run, so a changed overlay always wins.

## Verification run (clean room, published inputs only)

`rebuild_and_validate.sh` is what produced the published numbers below: it takes
only the three artefacts above (verified against `../src/SHA256SUMS`), rebuilds the
five libraries in a fresh local scratch tree, then runs the published 50-step
FL2VA pipeline plus `examples/minimax_h3_compare.py` with those libraries.

```
check:   SHA_OK snapshot.tgz / SHA_OK overlay.tar.gz / SHA_OK kernels.tgz
build:   BUILD_EXIT=0, FRESH_SO_COUNT=5
         libinfiniop.so 30,752,880 B    libinfinicore_cpp_api.so 13,297,376 B
         libinfinirt.so 48,184 B        libinfiniccl.so 52,104 B
         _infinicore.cpython-311-x86_64-linux-gnu.so 1,612,400 B
run:     RUN_RC=0, 50 steps, 18.2 s/step (910 s), out 1,619,081 B
compare: Size PASS / Length PASS / Average SSIM 79.907% (CMP_RC=0)
total:   960 s of wall clock for build + run + compare on 8 cards
```

This is the same 79-81% band as the released run (80.012%), from inputs that
contain no binary produced by this project.

## Why the scratch directory matters

The AFS mount used by the cluster answers `rename(2)` between directories with
`EXDEV` even inside a single mount.  Triton publishes every compiled object with
`os.replace()` from a temp directory, so the AOT step fails with

```
OSError: [Errno 18] Invalid cross-device link
```

Two independent countermeasures are in place: `patch_py.py` turns that specific
`os.replace` into copy + unlink, and step 3 puts `TMPDIR` / `TRITON_CACHE_DIR`
on a local scratch directory (`CRB_SCRATCH`, falling back to `mktemp -d`) instead
of the home directory.

## Expected result

```
build-work/ic/build/ninetoothed/            # 2 kernels + flash_attention.{c,h}
build-work/ic/build/linux/x86_64/release/
    libinfiniop.so  libinfinicore_cpp_api.so  libinfinirt.so
    libinfiniccl.so  _infinicore.cpython-311-x86_64-linux-gnu.so
```

Copy the five libraries over the ones shipped in the runtime image (or repack the
runtime tarball with them), then run `../jobs/run_tp8_fl2va.sh`: 50 denoising
steps at 768x928 / 124 frames with audio, followed by
`examples/minimax_h3_compare.py`.  A rebuild that matches the reported numbers
gives `Size PASS / Length PASS / Average SSIM` in the 79–81% band; a value
noticeably below that means the kernel grid or the device cache key did not come
out the same, and the two are the first things to check.
