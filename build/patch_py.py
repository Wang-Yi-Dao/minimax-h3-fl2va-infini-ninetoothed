#!/usr/bin/env python3
"""patch_py.py -- make the pip triton/ninetoothed wheels usable for an offline
(driverless) nine-toothed AOT build on DTK.

  usage: patch_py.py <dir that holds ninetoothed/ and triton/>

Five independent, named edits.  Each one is idempotent and prints
"PATCH <name> OK|SKIP|MISS" so a build log shows exactly which edits landed.
Every edit exists because the stock wheel assumes a live device, a CUDA-style
".c" temp file, or a rename(2) that the container filesystem does not provide.
"""
import pathlib
import sys

PY = pathlib.Path(sys.argv[1])


def edit(name, rel, pairs, marker):
    p = PY / rel
    if not p.is_file():
        print("PATCH %s MISS (no %s)" % (name, rel))
        return
    s = p.read_text()
    if marker in s:
        print("PATCH %s SKIP (already applied)" % name)
        return
    for old, new in pairs:
        if old not in s:
            print("PATCH %s MISS (anchor not found in %s)" % (name, rel))
            return
        s = s.replace(old, new, 1)
    p.write_text(s)
    print("PATCH %s OK" % name)


# 1. device properties -> fixed (warps, stages).  The stock helper asks the
#    driver for the current device, which does not exist in a batch job that
#    only compiles kernels.
edit("utils-default-config", "ninetoothed/utils.py", [(
    "def calculate_default_configs():\n",
    "def calculate_default_configs():\n"
    "    # DEFAULT_CONFIG: offline builds have no device to query\n"
    "    import os\n"
    "\n"
    "    override = os.environ.get(\"NINETOOTHED_DEFAULT_CONFIG\")\n"
    "    if override:\n"
    "        warps, stages = override.split(\",\")\n"
    "        return int(warps), int(stages)\n"
    "\n",
)], "DEFAULT_CONFIG")

# 2a. the target patch below needs sys (launch by interpreter path) and os (env).
def ensure_imports(rel, names):
    p = PY / rel
    s = p.read_text()
    add = [n for n in names if ("import %s\n" % n) not in s]
    if not add:
        print("PATCH aot-imports SKIP (already applied)")
        return
    block = "".join("import %s\n" % n for n in add)
    anchor = "import subprocess\n"
    s = s.replace(anchor, anchor + block, 1) if anchor in s else block + s
    p.write_text(s)
    print("PATCH aot-imports OK (%s)" % ",".join(add))


ensure_imports("ninetoothed/aot.py", ["sys", "os"])

# 2. forward an explicit --target to the triton compile subprocess.
edit("aot-target", "ninetoothed/aot.py", [(
    '        command = [\n            "python",\n            "-m",\n            "triton.tools.compile",\n            str(path),\n',
    "        command = [\n            sys.executable,\n            \"-m\",\n            \"triton.tools.compile\",\n        ]\n\n"
    "        # P29_TARGET: offline builds have no device, so forward an explicit target\n"
    "        _target = os.environ.get(\"NINETOOTHED_TARGET\")\n"
    "        if _target:\n            command += [\"--target\", _target]\n\n"
    "        command += [\n            str(path),\n",
)], "P29_TARGET")

# 3. the HIP back end emits "<kernel>.<hash>.cpp", not ".c".
edit("aot-source-ext", "ninetoothed/aot.py", [(
    '    c_source_file_name = f"{kernel_name}.{signature_hash}.c"\n',
    "    # HIPCPP_COMPAT: the hip template emits .cpp, the cuda template emits .c\n"
    "    _src_ext = \".cpp\" if any(n.endswith(\".cpp\") for n in output_contents) else \".c\"\n"
    "    c_source_file_name = f\"{kernel_name}.{signature_hash}{_src_ext}\"\n"
    "    if c_source_file_name not in output_contents:\n"
    "        _cands = [n for n in output_contents\n"
    "                  if n.startswith(f\"{kernel_name}.{signature_hash}.\") and n.endswith((\".c\", \".cpp\"))]\n"
    "        if not _cands:\n"
    "            raise KeyError(f\"no source for {kernel_name}.{signature_hash}: {sorted(output_contents)}\")\n"
    "        c_source_file_name = _cands[0]\n",
)], "HIPCPP_COMPAT")

# 4. keep the generated globals block in the emitted source.
edit("aot-globals", "ninetoothed/aot.py", [(
    '    kernel_start = c_source_file.find("//")\n',
    '    kernel_start = c_source_file.find("// globals")\n'
    '    if kernel_start < 0:\n'
    '        kernel_start = c_source_file.find("//")\n',
)], "// globals")

# 5. xmake compiles the emitted sources itself, so skip the per-kernel .so.
edit("aot-skip-lib", "ninetoothed/aot.py", [(
    "def _generate_launch_func(kernel_name, output_dir):\n"
    "    output_dir = pathlib.Path(output_dir)\n",
    "def _generate_launch_func(kernel_name, output_dir):\n"
    "    output_dir = pathlib.Path(output_dir)\n"
    "\n"
    "    if os.environ.get(\"NINETOOTHED_SKIP_LIB\") == \"1\":  # SKIP_LIB: xmake compiles the sources\n"
    "        return None\n"
    "\n",
)], "SKIP_LIB")

# 6. the python -> C type map lives on the driver; fall back to the back end
#    table when there is no device driver to ask.
edit("triton-ty-to-cpp", "triton/tools/compile.py", [(
    "    ty_to_cpp = triton.runtime.driver.active.map_python_to_cpp_type\n",
    "    # DRIVERLESS: do not touch the device driver when --target is given\n"
    "    try:\n"
    "        ty_to_cpp = triton.runtime.driver.active.map_python_to_cpp_type\n"
    "    except Exception:  # AMD_TY_TO_CPP: reuse the back end's authoritative map\n"
    "        from triton.backends.amd.driver import ty_to_cpp\n",
)], "DRIVERLESS")

# 7. this filesystem answers rename(2) with EXDEV even inside one mount, and
#    triton publishes every cache entry with os.replace().
edit("triton-cache-exdev", "triton/runtime/cache.py", [(
    "        os.replace(temp_path, filepath)\n        os.removedirs(temp_dir)\n",
    "        try:\n"
    "            os.replace(temp_path, filepath)\n"
    "        except OSError:\n"
    "            # EXDEV_FALLBACK: the container mount refuses cross-directory rename\n"
    "            import shutil as _shutil\n"
    "\n"
    "            _shutil.copyfile(temp_path, filepath)\n"
    "            try:\n"
    "                os.unlink(temp_path)\n"
    "            except OSError:\n"
    "                pass\n"
    "        try:\n"
    "            os.removedirs(temp_dir)\n"
    "        except OSError:\n"
    "            pass\n",
)], "EXDEV_FALLBACK")

# 8. triton 3.3.0 registers the short flag -w on more than one option
#    (--num-ldmatrixes, --enable-mmacfuse ...), so argparse aborts before the
#    kernel is even parsed.  Keep -w on --num-warps, drop it everywhere else.
CONFLICT_W = "ARG_W_DEDUP"
CMP_PY = PY / "triton" / "tools" / "compile.py"
if not CMP_PY.is_file():
    print("PATCH triton-argparse-w MISS (no triton/tools/compile.py)")
elif CONFLICT_W in CMP_PY.read_text():
    print("PATCH triton-argparse-w SKIP (already applied)")
else:
    _lines = CMP_PY.read_text().splitlines(True)
    _dropped = 0
    for _i, _line in enumerate(_lines):
        if '"-w"' in _line and "num-warps" not in _line:
            _new = _line.replace(', "-w"', "")
            if _new != _line:
                _lines[_i] = _new
                _dropped += 1
    CMP_PY.write_text("".join(_lines) + "# " + CONFLICT_W + "\n")
    print("PATCH triton-argparse-w OK (short -w dropped on %d options)" % _dropped)

print("PATCH_DONE")
