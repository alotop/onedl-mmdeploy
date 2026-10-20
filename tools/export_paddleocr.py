# Copyright (c) OpenMMLab. All rights reserved.
"""Export PaddleOCR inference models to mmdeploy-runnable TensorRT engines.

PaddleOCR publishes *inference* models (Paddle static graphs bundled with an
``inference.yml`` that documents the pre/post-processing), so no mmlab
training pipeline is involved.  The export path is:

1. fetch and extract a PaddleOCR inference model from the official model
   zoo (or reuse a local model directory / tarball);
2. translate the Paddle graph to ONNX with ``paddle2onnx``;
3. read the ONNX I/O names together with the TensorRT optimization profiles
   recorded in ``inference.yml``;
4. build a TensorRT engine through mmdeploy's ``onnx2tensorrt`` backend and
   dump a deploy config that describes the whole preprocessing /
   postprocessing contract of the model.

``paddle2onnx`` is ABI-locked to one paddlepaddle build (its C++ extension
links against ``libpaddle`` symbols), therefore the translation step runs in
a *separate* interpreter passed with ``--paddle-python``.  The engine build
itself runs in the current interpreter and only needs mmdeploy + TensorRT.

Outputs (all under ``--work-dir``):

- ``model/``              extracted PaddleOCR inference model
- ``<stem>.onnx``         translated ONNX model
- ``<stem>.engine``       TensorRT engine built by mmdeploy
- ``<stem>_tensorrt.py``  mmdeploy deploy config used for the conversion
- ``<stem>_meta.json``    preprocessing / postprocessing contract + I/O info
- ``<stem>_dict.json``    character table (recognition models only)

The engine is consumed with mmdeploy's TensorRT backend::

    from mmdeploy.backend.tensorrt import TRTWrapper
    engine = TRTWrapper('PP-OCRv5_mobile_det.engine')

Note that PaddleOCR models enter mmdeploy as ONNX, so the rewriter-based
end-to-end path (`mmdeploy.apis.inference`) does not apply: preprocessing and
the DB / CTC postprocessing stay in the application, and their parameters are
recorded in the generated deploy config and ``<stem>_meta.json``.

Examples:
    # official model name, fp32 engine with the profiles from inference.yml
    python tools/export_paddleocr.py PP-OCRv5_mobile_det \
        --paddle-python /path/to/paddle2onnx-env/python

    # fp16 engine, cap the dynamic width/height at 1920
    python tools/export_paddleocr.py PP-OCRv5_mobile_rec --fp16 \
        --max-side 1920 --paddle-python /path/to/paddle2onnx-env/python

    # already downloaded model directory
    python tools/export_paddleocr.py --model-dir /path/to/PP-OCRv5_mobile_det \
        --paddle-python /path/to/paddle2onnx-env/python
"""

import argparse
import json
import os
import os.path as osp
import pprint
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

DEFAULT_URL_TEMPLATE = (
    'https://paddle-model-ecology.bj.bcebos.com/paddlex/'
    'official_inference_model/paddle3.0.0/{name}_infer.tar')

#: default optimization profile (min / opt / max) used when a model does not
#: ship TensorRT shapes in ``inference.yml``.  ``None`` entries are derived
#: from the ONNX graph.
DEFAULT_PROFILES = {
    'det':
    dict(
        min_shape=[1, 3, 32, 32],
        opt_shape=[1, 3, 736, 736],
        max_shape=[1, 3, 1600, 1600]),
    'rec':
    dict(
        min_shape=[1, 3, 48, 32],
        opt_shape=[1, 3, 48, 320],
        max_shape=[1, 3, 48, 1280]),
    'cls':
    dict(
        min_shape=[1, 3, 80, 160],
        opt_shape=[1, 3, 80, 160],
        max_shape=[1, 3, 80, 160]),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export a PaddleOCR inference model to ONNX and build a '
        'TensorRT engine with mmdeploy.')
    parser.add_argument(
        'model',
        nargs='?',
        default=None,
        help='PaddleOCR model name from the official model zoo (e.g. '
        '`PP-OCRv5_mobile_det`), or a path to an extracted model directory / '
        'model tarball')
    parser.add_argument(
        '--model-dir',
        default=None,
        help='path to an already extracted PaddleOCR inference model. '
        'Overrides the positional argument')
    parser.add_argument(
        '--work-dir',
        default=None,
        help='directory for the model / ONNX / engine. Defaults to '
        '`work-dirs/paddleocr/<stem>`')
    parser.add_argument(
        '--task',
        default='auto',
        choices=['auto', 'det', 'rec', 'cls'],
        help='model family; `auto` reads it from `inference.yml` / the model '
        'name')
    parser.add_argument(
        '--url-template',
        default=DEFAULT_URL_TEMPLATE,
        help='download URL template for official models, `{name}` is '
        'replaced by the model name')
    parser.add_argument(
        '--no-download',
        action='store_true',
        help='never download; fail if the model is not available locally')
    parser.add_argument(
        '--no-proxy',
        action='store_true',
        help='ignore `http_proxy` / `https_proxy` while downloading the model '
        '(the PaddleOCR model zoo is a public CDN, direct connections often '
        'work better than a local proxy)')
    parser.add_argument(
        '--paddle-python',
        default=None,
        help='interpreter that has a paddle2onnx build matching its '
        'paddlepaddle (falls back to $PADDLE2ONNX_PYTHON, then to the '
        'current interpreter)')
    parser.add_argument(
        '--opset',
        type=int,
        default=16,
        help='ONNX opset version passed to paddle2onnx')
    parser.add_argument(
        '--optimize-tool',
        default='onnxoptimizer',
        choices=['onnxoptimizer', 'polygraphy', 'none'],
        help='ONNX optimization used by paddle2onnx after the translation')
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='build the TensorRT engine in fp16 mode (the ONNX itself stays '
        'fp32; fp16 conversion is done by the TensorRT builder)')
    parser.add_argument(
        '--static',
        action='store_true',
        help='build a static-shape engine from the optimal profile instead '
        'of the dynamic min/opt/max profiles')
    parser.add_argument(
        '--max-side',
        type=int,
        default=1920,
        help='cap the dynamic height/width of the optimization profile. '
        'PaddleOCR resizes the long side to <=960 by default, so 1920 is a '
        'safe default; pass 0 to keep the (up to 4000px) profile recorded in '
        '`inference.yml`')
    parser.add_argument(
        '--max-workspace',
        type=int,
        default=1 << 30,
        help='TensorRT max workspace size in bytes')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for the engine build')
    return parser.parse_args()


def _stem(name: str) -> str:
    """Return a filesystem-safe stem for a model name/path."""
    name = name.rstrip('/\\')
    stem = osp.splitext(osp.basename(name))[0]
    # the official tarballs are named `<model>_infer.tar`
    return stem[:-len('_infer')] if stem.endswith('_infer') else stem


def _download(url: str, dst: str, retries: int = 6, use_proxy: bool = True):
    """Download `url` to `dst`, resuming partial files on flaky links."""
    print(f'      downloading {url}', flush=True)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(None if use_proxy else {}))
    total = 0
    for attempt in range(1, retries + 1):
        start = osp.getsize(dst) if osp.isfile(dst) else 0
        headers = {'User-Agent': 'mmdeploy'}
        if start:
            headers['Range'] = f'bytes={start}-'
        try:
            req = urllib.request.Request(url, headers=headers)
            with opener.open(req, timeout=60) as resp:
                partial = resp.getcode() == 206
                length = int(resp.headers.get('Content-Length') or 0)
                total = length + start if partial and length else length
                mode = 'ab' if partial else 'wb'
                done, last = (start if partial else 0), time.time()
                with open(dst, mode) as f:
                    while True:
                        chunk = resp.read(1 << 16)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 3:
                            last = time.time()
                            pct = f'{done / total * 100:5.1f}%' \
                                if total else ''
                            print(
                                f'        {done / 2**20:7.1f} MB {pct}',
                                flush=True)
        except Exception as exc:  # noqa: BLE001 - network errors
            print(f'      attempt {attempt} failed: {exc}', flush=True)
        size = osp.getsize(dst) if osp.isfile(dst) else 0
        if total and size >= total:
            break
        if attempt == retries:
            raise RuntimeError(f'failed to download {url} after {retries} '
                               f'attempts ({size} of {total} bytes)')
    print(f'      saved {dst} ({osp.getsize(dst) / 2**20:.1f} MB)')


def _extract_tar(tar_path: str, dst_dir: str):
    """Extract a model tarball, dropping the top-level directory."""
    tmp = dst_dir + '_extract'
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(tmp)
    entries = os.listdir(tmp)
    src = osp.join(tmp, entries[0]) if len(entries) == 1 and osp.isdir(
        osp.join(tmp, entries[0])) else tmp
    shutil.rmtree(dst_dir, ignore_errors=True)
    shutil.move(src, dst_dir)
    if osp.isdir(tmp):
        shutil.rmtree(tmp, ignore_errors=True)


def _find_model_files(model_dir: str):
    """Return ``(graph_file, params_file)`` of a Paddle inference model."""
    candidates = [
        ('inference.json', 'inference.pdiparams'),
        ('inference.pdmodel', 'inference.pdiparams'),
    ]
    for graph, params in candidates:
        if osp.isfile(osp.join(model_dir, graph)):
            params_path = osp.join(model_dir, params)
            return graph, params if osp.isfile(params_path) else None
    graphs = [
        f for f in os.listdir(model_dir) if f.endswith(('.json', '.pdmodel'))
    ]
    if len(graphs) == 1:
        params = [f for f in os.listdir(model_dir) if f.endswith('.pdiparams')]
        return graphs[0], params[0] if len(params) == 1 else None
    raise RuntimeError(
        f'No Paddle inference model found in `{model_dir}`: expected '
        '`inference.json` (paddle 3.x) or `inference.pdmodel` (paddle 2.x)')


def resolve_model(args, work_dir: str, model_dir: str) -> str:
    """Return the directory of an extracted PaddleOCR inference model."""
    if args.model_dir:
        model_dir = osp.abspath(args.model_dir)
        if not osp.isdir(model_dir):
            raise FileNotFoundError(f'`{model_dir}` is not a directory')
        return model_dir

    if osp.isdir(args.model):
        return osp.abspath(args.model)

    if osp.isfile(args.model) and tarfile.is_tarfile(args.model):
        _extract_tar(args.model, model_dir)
        return model_dir

    stem = _stem(args.model)
    if osp.isdir(model_dir) and osp.listdir(model_dir):
        print(f'[1/5] reusing model directory {model_dir}')
        return model_dir
    if args.no_download:
        raise FileNotFoundError(
            f'`{args.model}` is not a local path and `--no-download` was '
            f'given (expected model files in `{model_dir}`)')

    print(f'[1/5] downloading {stem} from the PaddleOCR model zoo ...')
    os.makedirs(model_dir, exist_ok=True)
    tar_path = osp.join(work_dir, f'{stem}_infer.tar')
    if not osp.isfile(tar_path) or osp.getsize(tar_path) < 1024:
        _download(
            args.url_template.format(name=stem),
            tar_path,
            use_proxy=not args.no_proxy)
    _extract_tar(tar_path, model_dir)
    return model_dir


def _load_yaml(path: str):
    """Load `inference.yml` (returns None when it is not available)."""
    if not osp.isfile(path):
        return None
    try:
        import yaml
    except ImportError:  # pragma: no cover - mmengine ships pyyaml
        return None
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def detect_task(model_name: str, yml) -> str:
    """Infer the model family from `inference.yml` or the model name."""
    if yml:
        post = yml.get('PostProcess') or {}
        name = str(post.get('name', '')).lower()
        if 'db' in name:
            return 'det'
        if 'ctc' in name or 'recogn' in name:
            return 'rec'
        if 'topk' in name or 'cls' in name:
            return 'cls'
        if isinstance(post.get('Topk'), dict):
            return 'cls'
    lowered = model_name.lower()
    if 'textline_ori' in lowered or lowered.endswith('_cls'):
        return 'cls'
    if lowered.endswith('_rec'):
        return 'rec'
    if lowered.endswith('_det'):
        return 'det'
    raise RuntimeError(
        f'Cannot infer the task of `{model_name}`; please pass --task')


def _setup_paddle_python(args) -> str:
    """Pick the interpreter used to run paddle2onnx."""
    candidates = [
        args.paddle_python,
        os.environ.get('PADDLE2ONNX_PYTHON'),
        sys.executable,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if not osp.isfile(candidate):
            # allow a bare command name resolved through PATH
            resolved = shutil.which(candidate)
            if resolved is None:
                continue
            candidate = resolved
        probe = subprocess.run([
            candidate, '-c', 'import paddle, paddle2onnx; '
            'print(paddle.__version__)'
        ],
                               capture_output=True,
                               text=True)
        if probe.returncode == 0:
            print(f'      paddle2onnx interpreter: {candidate} '
                  f'(paddle {probe.stdout.strip()})')
            return candidate
        err = (probe.stderr or '').strip().splitlines()
        print(f'      skip {candidate}: {err[-1] if err else "import failed"}')
    raise RuntimeError(
        'No interpreter with a usable paddle2onnx found.  paddle2onnx is '
        'ABI-locked to a paddlepaddle build, so install it in its own '
        'environment (`pip install paddlepaddle==<ver> paddle2onnx`) and '
        'pass `--paddle-python /path/to/that/python`.')


def convert_to_onnx(paddle_python: str, model_dir: str, graph_file: str,
                    params_file: str, onnx_path: str, args):
    """Translate the Paddle inference graph to ONNX with paddle2onnx."""
    model_filename = osp.join(model_dir, graph_file)
    params_filename = osp.join(model_dir, params_file) if params_file else ''
    optimize_tool = None if args.optimize_tool == 'none' \
        else args.optimize_tool
    code = ('import paddle2onnx; '
            'paddle2onnx.export('
            f'model_filename={model_filename!r}, '
            f'params_filename={params_filename!r}, '
            f'save_file={onnx_path!r}, '
            f'opset_version={args.opset}, '
            'auto_upgrade_opset=True, verbose=False, '
            'enable_onnx_checker=True, '
            f'optimize_tool={optimize_tool!r})')
    subprocess.run([paddle_python, '-c', code], check=True)


def _onnx_io_info(onnx_path: str):
    """Read input/output names and shapes from an ONNX model.

    Returns:
        (inputs, outputs) mapping tensor name to a list of dimensions.
        Dynamic dimensions are kept as their symbolic name.
    """
    import onnx

    model = onnx.load(onnx_path)

    def dims_of(tensor):
        out = []
        for dim in tensor.type.tensor_type.shape.dim:
            out.append(dim.dim_param if dim.dim_param else dim.dim_value)
        return out

    inputs = {t.name: dims_of(t) for t in model.graph.input}
    outputs = {t.name: dims_of(t) for t in model.graph.output}
    return inputs, outputs


def _dynamic_axes(inputs, outputs):
    """Build the mmdeploy-style ``{tensor: {axis: name}}`` description."""
    axes = {}
    for name, dims in list(inputs.items()) + list(outputs.items()):
        dyn = {i: d for i, d in enumerate(dims) if isinstance(d, str)}
        if dyn:
            axes[name] = dyn
    return axes


def _profiles_from_yml(yml, input_names):
    """Read the TensorRT min/opt/max shapes recorded in `inference.yml`."""
    if not yml:
        return None
    backend = ((yml.get('Hpi') or {}).get('backend_configs')
               or {}).get('tensorrt') or {}
    shapes = backend.get('dynamic_shapes')
    if not shapes:
        return None
    profiles = {}
    for name in input_names:
        entry = shapes.get(name)
        if not entry or len(entry) != 3:
            continue
        shapes_i = [[int(v) for v in e] for e in entry]
        if any(v <= 0 for shape in shapes_i for v in shape):
            # a dynamic (-1) entry cannot be used as a builder profile
            continue
        profiles[name] = dict(
            min_shape=shapes_i[0],
            opt_shape=shapes_i[1],
            max_shape=shapes_i[2])
    return profiles or None


def _fallback_profile(task, dims):
    """Derive a profile for models without TensorRT shapes in the yml."""
    if task not in DEFAULT_PROFILES:
        raise RuntimeError(f'No default profile for task `{task}`')
    profile = {k: list(v) for k, v in DEFAULT_PROFILES[task].items()}
    # honour static dimensions of the graph (e.g. the 48px rec height)
    for i, dim in enumerate(dims):
        if isinstance(dim, int):
            for key in profile:
                profile[key][i] = dim
    return profile


def _clip_profile(profile, max_side):
    """Cap the height/width axes of a profile at `max_side` pixels."""
    if not max_side:
        return profile
    out = {}
    for key, shape in profile.items():
        shape = list(shape)
        for axis in (2, 3):
            if axis < len(shape):
                shape[axis] = min(shape[axis], max_side)
        out[key] = shape
    # keep min <= opt <= max after the clamp
    keys = ['min_shape', 'opt_shape', 'max_shape']
    for axis in (2, 3):
        values = [out[k][axis] for k in keys]
        values = sorted(values)
        for key, value in zip(keys, values):
            out[key][axis] = value
    return out


def build_profiles(task, inputs, yml, args):
    """Return the TensorRT optimization profiles per input tensor."""
    from_yml = _profiles_from_yml(yml, inputs)
    profiles = {}
    for name, dims in inputs.items():
        profile = (from_yml or {}).get(name)
        if profile is None:
            profile = _fallback_profile(task, dims)
        profile = _clip_profile(profile, args.max_side)
        if args.static:
            opt = profile['opt_shape']
            profile = dict(
                min_shape=list(opt), opt_shape=list(opt), max_shape=list(opt))
        # a static ONNX dimension cannot be overridden by the profile
        for axis, dim in enumerate(dims):
            if isinstance(dim, int):
                for key in profile:
                    profile[key][axis] = dim
        profiles[name] = profile
    return profiles


def _fmt_literal(obj, indent: str = ' ' * 8) -> str:
    """Format a nested literal and keep its continuation lines indented."""
    text = pprint.pformat(obj, width=70, sort_dicts=False)
    return text.replace('\n', '\n' + indent)


def _dump_deploy_cfg(path, stem, task, args, inputs, outputs, profiles, yml):
    """Write the mmdeploy deploy config describing the conversion."""
    dynamic_axes = _dynamic_axes(inputs, outputs)
    pre = (yml or {}).get('PreProcess', {}).get('transform_ops')
    post = (yml or {}).get('PostProcess')
    lines = [
        '# Auto-generated by tools/export_paddleocr.py -- do not edit by '
        'hand.',
        f'# Model: {stem} (task={task})',
        f'# fp16_mode={args.fp16}, static={args.static}, '
        f'opset={args.opset}',
        '',
        'onnx_config = dict(',
        "    type='onnx',",
        '    export_params=True,',
        '    keep_initializers_as_inputs=False,',
        f'    save_file={stem + ".onnx"!r},',
        f'    input_names={list(inputs)!r},',
        f'    output_names={list(outputs)!r},',
        '    input_shape=None,',
    ]
    if dynamic_axes:
        lines.append(f'    dynamic_axes={dynamic_axes!r},')
    lines += ['    optimize=True)', '']
    lines += [
        'codebase_config = dict(',
        "    type='paddleocr',",
        f'    model_type={task!r},',
    ]
    if pre:
        lines.append('    preprocess=' + _fmt_literal(pre) + ',')
    if post:
        post = dict(post)
        # the character table is huge and stored in `<stem>_meta.json`
        post.pop('character_dict', None)
        lines.append('    postprocess=' + _fmt_literal(post) + ',')
    lines += [
        ')', '', 'backend_config = dict(', "    type='tensorrt',",
        '    common_config=dict(', f'        fp16_mode={args.fp16},',
        f'        max_workspace_size={args.max_workspace}),',
        '    model_inputs=[dict(', f'        input_shapes={profiles!r})])', ''
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _write_dict_file(work_dir, stem, post):
    """Dump the recognition character table and return its path/count."""
    chars = (post or {}).get('character_dict')
    if not chars:
        return None, None
    # the table is taken verbatim from inference.yml; keep it as JSON so that
    # whitespace-only entries survive a round trip
    path = osp.join(work_dir, f'{stem}_dict.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(chars, f, ensure_ascii=False)
    return path, len(chars)


def main():
    args = parse_args()
    if args.model is None and args.model_dir is None:
        raise SystemExit('either a model name or --model-dir is required')

    name = args.model or args.model_dir
    stem = _stem(name)
    work_dir = args.work_dir or osp.join('work-dirs', 'paddleocr', stem)
    work_dir = osp.abspath(work_dir)
    model_dir = osp.join(work_dir, 'model')
    os.makedirs(work_dir, exist_ok=True)

    model_dir = resolve_model(args, work_dir, model_dir)
    graph_file, params_file = _find_model_files(model_dir)
    yml = _load_yaml(osp.join(model_dir, 'inference.yml'))
    if yml:
        stem = (yml.get('Global') or {}).get('model_name') or stem
    task = args.task if args.task != 'auto' else detect_task(stem, yml)
    print(f'      model dir: {model_dir}')
    print(f'      task: {task}, graph: {graph_file}')

    print('[2/5] translating the Paddle graph to ONNX ...')
    paddle_python = _setup_paddle_python(args)
    onnx_path = osp.join(work_dir, f'{stem}.onnx')
    convert_to_onnx(paddle_python, model_dir, graph_file, params_file,
                    onnx_path, args)
    print(f'      onnx: {onnx_path}')

    print('[3/5] inspecting ONNX I/O ...')
    inputs, outputs = _onnx_io_info(onnx_path)
    profiles = build_profiles(task, inputs, yml, args)
    print(f'      inputs={inputs}')
    print(f'      outputs={list(outputs)}')
    for tname, profile in profiles.items():
        print(f'      profile[{tname}]: min={profile["min_shape"]} '
              f'opt={profile["opt_shape"]} max={profile["max_shape"]}')

    print('[4/5] building the TensorRT engine with mmdeploy ...')
    cfg_path = osp.join(work_dir, f'{stem}_tensorrt.py')
    _dump_deploy_cfg(cfg_path, stem, task, args, inputs, outputs, profiles,
                     yml)

    from mmdeploy.apis.tensorrt import onnx2tensorrt
    engine_name = f'{stem}.engine'
    onnx2tensorrt(
        work_dir, engine_name, 0, cfg_path, onnx_path, device=args.device)
    engine_path = osp.join(work_dir, engine_name)
    print(f'      engine: {engine_path}')

    print('[5/5] writing meta ...')
    post = dict((yml or {}).get('PostProcess') or {})
    dict_path, num_chars = _write_dict_file(work_dir, stem, post)
    if 'character_dict' in post:
        post['character_dict'] = dict(
            file=dict_path, num_chars=num_chars) if dict_path else None
    meta = dict(
        model=stem,
        task=task,
        source=args.model or args.model_dir,
        paddle_python=paddle_python,
        opset=args.opset,
        fp16=bool(args.fp16),
        static=bool(args.static),
        device=args.device,
        input_names=list(inputs),
        output_names=list(outputs),
        input_dims=inputs,
        output_dims=outputs,
        dynamic_axes=_dynamic_axes(inputs, outputs),
        input_shapes=profiles,
        preprocess=(yml or {}).get('PreProcess'),
        postprocess=post,
        onnx=onnx_path,
        engine=engine_path,
        engine_size_mb=round(osp.getsize(engine_path) / 2**20, 2),
        deploy_config=cfg_path)
    meta_path = osp.join(work_dir, f'{stem}_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f'      meta: {meta_path}')
    print(f'export finished: {engine_path}')


if __name__ == '__main__':
    main()
