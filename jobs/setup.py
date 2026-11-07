import os, sys, json, glob, zipfile, tarfile, time, shutil, traceback

W = sys.argv[1]
AFS = sys.argv[2]
SH = sys.argv[3]
H = os.path.join(AFS, 'minimax_h3')
os.chdir(W)
T = time.time()


def log(*a):
    print('[%6.1f]' % (time.time() - T), *a, flush=True)


def restore_tars():
    for name in ('infinicore.tgz', 'infinilm.tgz'):
        p = os.path.join(H, 'artifacts', name)
        t0 = time.time()
        with tarfile.open(p) as t:
            t.extractall(W, filter='fully_trusted')
        log('restored', name, round(time.time() - t0, 1), 's')
    with tarfile.open(os.path.join(H, 'src', 'InfiniLM.tgz')) as t:
        t.extractall(W, filter='fully_trusted')
    log('restored src/InfiniLM.tgz')
    log('TOP', sorted(x for x in os.listdir(W) if not x.startswith('logs'))[:20])
    log('infini_lib', os.path.isdir(os.path.join(W, '.infini', 'lib')),
        'ic_build', os.path.isdir(os.path.join(W, 'InfiniCore', 'build')),
        'ilm_examples', os.path.exists(os.path.join(W, 'InfiniLM', 'examples', 'minimax_h3.py')))


def wheels():
    dst = os.path.join(W, 'pylibs')
    os.makedirs(dst, exist_ok=True)
    used = []
    for w in sorted(glob.glob(os.path.join(H, 'wheels', '*.whl'))):
        b = os.path.basename(w)
        if 'flash_attn' in b or b.startswith('torchvision'):
            continue
        if b.startswith('transformers-5.0.0'):
            continue
        with zipfile.ZipFile(w) as z:
            z.extractall(dst)
        used.append(b)
    log('wheels', len(used), [u for u in used if u.startswith('transformers')])
    for name in ('flash_attn', 'torchvision', 'torchvision.libs'):
        shutil.rmtree(os.path.join(dst, name), ignore_errors=True)
    for p in glob.glob(os.path.join(dst, 'torchvision-*.dist-info')):
        shutil.rmtree(p, ignore_errors=True)


def flash_attn_so():
    whl = os.path.join(H, 'wheels5', 'fa_dtk2604_torch290_cp311.whl')
    fa = os.path.join(W, 'fa')
    os.makedirs(fa, exist_ok=True)
    dst = os.path.join(W, 'pylibs')
    with zipfile.ZipFile(whl) as z:
        names = z.namelist()
        sos = [n for n in names if n.endswith('.so')]
        for n in sos:
            z.extract(n, fa)
        for n in names:
            if n.startswith('flash_attn/') and not n.endswith('/'):
                z.extract(n, dst)
    sos = [os.path.join(fa, s) for s in sos]
    log('fa so', [(os.path.basename(s), os.path.getsize(s)) for s in sos])
    # transformers imports flash_attn eagerly when it is importable; that package needs einops
    # which is not shipped here, so keep only the .so and drop the python package.
    shutil.rmtree(os.path.join(dst, 'flash_attn'), ignore_errors=True)
    for p in glob.glob(os.path.join(dst, 'flash_attn*.dist-info')):
        shutil.rmtree(p, ignore_errors=True)
    log('flash_attn python package removed:',
        not os.path.exists(os.path.join(dst, 'flash_attn')))
    return sos[0] if sos else ''


def model_farm():
    root = os.path.join(W, 'model')
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    for name in sorted(os.listdir(SH)):
        if name == 'processor':
            continue
        os.symlink(os.path.join(SH, name), os.path.join(root, name))
    dst = os.path.join(root, 'processor')
    shutil.copytree(os.path.join(SH, 'processor'), dst)
    cfg_path = os.path.join(dst, 'preprocessor_config.json')
    cfg = json.load(open(cfg_path))
    cfg['image_processor_type'] = 'Qwen2VLImageProcessor'
    json.dump(cfg, open(cfg_path, 'w'), indent=2)
    log('model farm ready, image_processor_type ->', cfg['image_processor_type'])


def patch_transformers():
    PU = os.path.join(W, 'pylibs', 'transformers', 'processing_utils.py')
    s = open(PU).read()
    if 'FL2VA_SKIP_VIDEO' not in s:
        old = """            elif is_primary:
                # Primary non-tokenizer sub-processor: load via Auto class
                auto_processor_class = MODALITY_TO_AUTOPROCESSOR_MAPPING[sub_processor_type]
"""
        new = """            elif is_primary:
                # Primary non-tokenizer sub-processor: load via Auto class
                from .utils import is_torchvision_available as _itva  # FL2VA_SKIP_VIDEO
                if sub_processor_type == "video_processor" and not _itva():
                    args.append(None)
                    continue
                auto_processor_class = MODALITY_TO_AUTOPROCESSOR_MAPPING[sub_processor_type]
"""
        assert old in s, 'processing_utils anchor missing'
        s = s.replace(old, new, 1)
    if 'FL2VA_NULL_VIDEO' not in s:
        old2 = """        if not isinstance(argument, proper_class):
            raise TypeError(
                f"Received a {type(argument).__name__} for argument {argument_name}, but a {class_name} was expected."
            )
"""
        new2 = """        if argument is None and argument_name == "video_processor":  # FL2VA_NULL_VIDEO
            return proper_class
        if not isinstance(argument, proper_class):
            raise TypeError(
                f"Received a {type(argument).__name__} for argument {argument_name}, but a {class_name} was expected."
            )
"""
        assert old2 in s, 'check_argument anchor missing'
        s = s.replace(old2, new2, 1)
    open(PU, 'w').write(s)
    log('processing_utils patched')
    V = os.path.join(W, 'pylibs', 'transformers', 'models', 'auto', 'video_processing_auto.py')
    v = open(V).read()
    v = v.replace('if class_name in extractors:', 'if extractors is not None and class_name in extractors:')
    v = v.replace('elif video_processor_class is not None:', 'elif video_processor_class is not None and is_torchvision_available():')
    v = v.replace('elif type(config) in VIDEO_PROCESSOR_MAPPING:', 'elif is_torchvision_available() and type(config) in VIDEO_PROCESSOR_MAPPING:')
    a = '@requires(backends=("vision", "torchvision"))'
    if a in v:
        v = v.replace(a, '@requires(backends=("vision",))')
    b = """    # If the torchvision is not available, we set it to None
    if not is_torchvision_available():
        fast_video_processor_class = None
"""
    if b in v:
        v = v.replace(b, """    # FL2VA: keep mapping without torchvision
    if False:
        fast_video_processor_class = None
""")
    open(V, 'w').write(v)
    log('video_processing_auto patched')


def patch_diffusers():
    P = os.path.join(W, 'pylibs', 'diffusers', 'modular_pipelines', 'minimax_h3', 'encoders.py')
    s = open(P).read()
    if 'FL2VA_MM_IDS' not in s:
        old = "    mm_token_type_ids = torch.tensor(processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device)\n"
        new = """    # FL2VA_MM_IDS: transformers has no create_mm_token_type_ids; derive like Qwen3-VL does internally
    _ids = np.asarray([token_ids], dtype=np.int64)
    _mm = np.zeros_like(_ids)
    _img = getattr(processor, "image_token_id", None)
    _vid = getattr(processor, "video_token_id", None)
    if _img is not None:
        _mm[_ids == _img] = 1
    if _vid is not None:
        _mm[_ids == _vid] = 2
    mm_token_type_ids = torch.tensor(_mm, dtype=torch.long, device=device)
"""
        assert old in s, 'encoders anchor missing'
        s = s.replace(old, new, 1)
        open(P, 'w').write(s)
    log('encoders patched:', 'FL2VA_MM_IDS' in open(P).read())


steps = [
    ('tars', restore_tars),
    ('wheels', wheels),
    ('fa', flash_attn_so),
    ('model', model_farm),
    ('patch_tf', patch_transformers),
    ('patch_df', patch_diffusers),
]
for label, fn in steps:
    try:
        r = fn()
        if label == 'fa':
            open(os.path.join(W, 'fa_so_path.txt'), 'w').write(r or '')
    except Exception:
        log('STEP_FAIL', label)
        traceback.print_exc()
        sys.exit(21)
log('SETUP_OK')
