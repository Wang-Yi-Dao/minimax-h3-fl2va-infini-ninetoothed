# gen_dispatch.py -- emit ic/build/ninetoothed/flash_attention.{c,h} for the two
# deployed flash-attention grids (emb_dim 72 and 128, bf16, block 256x64, no kv
# cache / no causal mask / no attn mask, LOWER_RIGHT causal variant).
#
# The generated dispatcher is what src/infiniop/ops/flash_attention/ninetoothed/
# descriptor.h includes, and its body is a chain of `if <condition> return
# launch_<kernel>(...);`, so the C++ side sees only one entry point.
#
# Run inside the tree:  cd $W/ic && PYTHONPATH=$W/py:$W/ic/src python3 gen_dispatch.py
import inspect
import ninetoothed
from ninetoothed.aot import _HEADER_PATH
from infiniop.ninetoothed.build import (BUILD_DIRECTORY_PATH, _generate_condition, _generate_suffix)
from infiniop.ops.flash_attention.ninetoothed import flash_attention as FA

combos = []
for emb in (72, 128):
    combos.append(dict(with_kv_cache=0, emb_dim=emb, is_causal=0, with_attn_mask=0,
                       causal_variant=FA.CausalVariant.LOWER_RIGHT, dtype=ninetoothed.bfloat16,
                       block_size_m=256, block_size_n=64))

headers, app_names, combinations, launches = [], None, [], []
for comb in combos:
    arrangement, application, tensors = FA.premake(**comb)
    c = dict(comb)
    for k, v in c.items():
        if isinstance(v, str):
            c[k] = 'INFINI_DTYPE_' + v.replace('fp', 'F').upper()
    c = {f'{k}_': v for k, v in c.items()}
    kernel_name = 'flash_attention_' + _generate_suffix(c.values())
    names = ('stream',) + tuple(inspect.signature(application).parameters.keys())
    if app_names is None:
        app_names = names
    elif app_names != names:
        raise SystemExit('application signatures differ')
    headers.append(BUILD_DIRECTORY_PATH / f'{kernel_name}.h')
    combinations.append(c)
    launches.append('    if (%s)\n        return launch_%s(%s);' % (_generate_condition(c), kernel_name, ', '.join(names)))

merged = {}
for c in combinations:
    merged.update(c)
# keep merged order deterministic: keys as declared in combos[0]
param_names = list(app_names) + [k for k in combinations[0] if k in merged]
param_types = ['NineToothedStream'] + ['NineToothedTensor' for _ in range(len(app_names) - 1)] + ['int'] * len(param_names[len(app_names):])
assert len(set(param_names)) == len(param_names), 'duplicate parameter names: %s' % param_names
includes = '\n'.join('#include \"%s\"' % h for h in headers)
param_decls = ', '.join('%s %s' % (t, n) for t, n in zip(param_types, param_names))
func_sig = 'NineToothedResult launch_flash_attention(%s)' % param_decls
op_decl = '#ifdef __cplusplus\nextern \"C\" {\n%s;\n}\n#else\n%s;\n#endif' % (func_sig, func_sig)
op_def = '%s {\n%s\n    return INFINI_STATUS_NOT_IMPLEMENTED;\n}' % (func_sig, '\n'.join(launches))
(BUILD_DIRECTORY_PATH / 'flash_attention.c').write_text('#include \"flash_attention.h\"\n\n#include \"infinicore.h\"\n\n%s\n\n%s\n' % (includes, op_def))
(BUILD_DIRECTORY_PATH / 'flash_attention.h').write_text('#include \"%s\"\n\n%s\n' % (_HEADER_PATH, op_decl))
print('DISPATCH_OK nparams=%d' % len(param_names))
