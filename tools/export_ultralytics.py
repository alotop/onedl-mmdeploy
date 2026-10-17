# Copyright (c) OpenMMLab. All rights reserved.
"""Export ultralytics checkpoints to mmdeploy-runnable TensorRT engines.

This tool bridges the ultralytics model zoo
(https://github.com/ultralytics/ultralytics) and mmdeploy's deployment
pipeline:

1. load an ultralytics checkpoint (``*.pt``; downloaded automatically into
   the work directory when a model name is given instead of a local path);
2. export it to ONNX with the ultralytics exporter, keeping the I/O names
   and dynamic axes that mmdeploy expects;
3. build a TensorRT engine from that ONNX through mmdeploy's
   ``onnx2tensorrt`` converter (fp16 mode / workspace / optimization
   profiles are all handled by mmdeploy's TensorRT backend).

No mmlab training pipeline is involved: any stock ultralytics checkpoint
can be exported directly.

Outputs (all under ``--work-dir``):

- ``<stem>.pt``            downloaded checkpoint (kept for reproducibility)
- ``<stem>.onnx``          exported ONNX model
- ``<stem>.engine``        TensorRT engine built by mmdeploy
- ``<stem>_tensorrt.py``   mmdeploy deploy config used for the conversion
- ``<stem>_meta.json``     machine-readable summary of the export

The engine can be consumed with mmdeploy's TensorRT backend, e.g.::

    from mmdeploy.backend.tensorrt import TRTWrapper
    engine = TRTWrapper('yolov8n.engine')

Examples:
    # static-shape fp16 engine for yolov8n
    python tools/export_ultralytics.py yolov8n.pt --fp16

    # dynamic input size (320..1280, opt 640)
    python tools/export_ultralytics.py yolo11n-seg.pt --dynamic --fp16

    # local checkpoint
    python tools/export_ultralytics.py /path/to/weights.pt \
        --work-dir work-dirs/ultralytics/custom
"""

import argparse
import json
import os
import os.path as osp
import shutil
import sys

# Never let ultralytics auto-install missing dependencies: a silent
# `pip install` from inside an export can uninstall/replace packages of
# the active environment.  Models that need extra packages (e.g. YOLO-NAS
# needs `super_gradients`) will fail with an explicit error instead.
os.environ.setdefault('YOLO_AUTOINSTALL', 'False')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export an ultralytics checkpoint to ONNX and build a '
        'TensorRT engine with mmdeploy.')
    parser.add_argument(
        'model',
        help='ultralytics model name (e.g. `yolov8n.pt`; downloaded into the '
        'work dir automatically) or a path to a local checkpoint')
    parser.add_argument(
        '--work-dir',
        default=None,
        help='directory to store the checkpoint / ONNX / engine. Defaults '
        'to `work-dirs/ultralytics/<model stem>`')
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='input resolution used for the export (square images)')
    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='export with dynamic height/width axes and build the engine '
        'with an optimization profile covering [min-imgsz, max-imgsz]')
    parser.add_argument(
        '--min-imgsz',
        type=int,
        default=320,
        help='lower bound of the dynamic input size (with --dynamic)')
    parser.add_argument(
        '--max-imgsz',
        type=int,
        default=1280,
        help='upper bound of the dynamic input size (with --dynamic)')
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='build the TensorRT engine in fp16 mode (the ONNX itself stays '
        'fp32; fp16 conversion is done by the TensorRT builder)')
    parser.add_argument(
        '--opset',
        type=int,
        default=None,
        help='ONNX opset version. By default the opset chosen by the '
        'ultralytics exporter is used')
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='run the ultralytics ONNX simplifier (onnxslim) after export')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='device used for the ONNX export and the engine build')
    parser.add_argument(
        '--max-workspace',
        type=int,
        default=1 << 30,
        help='TensorRT max workspace size in bytes')
    parser.add_argument(
        '--ultralytics-src',
        default=None,
        help='optional path prepended to sys.path so a local clone of the '
        'ultralytics repository is used instead of the installed package')
    return parser.parse_args()


def _stem(model_arg: str) -> str:
    """Return a filesystem-safe stem for the model name/path."""
    return osp.splitext(osp.basename(model_arg))[0]


def _setup_ultralytics(args):
    """Import the ultralytics package (optionally from a local clone)."""
    if args.ultralytics_src:
        sys.path.insert(0, osp.abspath(args.ultralytics_src))

    import ultralytics
    from ultralytics import YOLO
    return ultralytics, YOLO


def _export_onnx(args, work_dir: str) -> str:
    """Download the checkpoint (if needed) and export it to ONNX.

    The export happens with the work dir as the current directory, so both
    the downloaded ``.pt`` and the exported ``.onnx`` land there.
    """
    _, YOLO = _setup_ultralytics(args)

    model_arg = args.model
    if osp.isfile(model_arg):
        model_arg = osp.abspath(model_arg)

    prev_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        model = YOLO(model_arg)
        export_kwargs = dict(
            format='onnx',
            imgsz=args.imgsz,
            dynamic=args.dynamic,
            half=False,  # fp32 ONNX; fp16 is decided by the TRT builder
            simplify=args.simplify,
            device=args.device,
            batch=1)
        if args.opset is not None:
            export_kwargs['opset'] = args.opset
        onnx_path = model.export(**export_kwargs)
        # the exporter returns a path relative to the cwd it ran in --
        # resolve it while we are still inside the work dir
        if isinstance(onnx_path, (list, tuple)):
            onnx_path = onnx_path[0]
        onnx_path = osp.abspath(onnx_path)
        task = model.task
        names = model.names
    finally:
        os.chdir(prev_cwd)

    dst = osp.join(work_dir, f'{_stem(args.model)}.onnx')
    if osp.abspath(dst) != onnx_path:
        shutil.copy2(onnx_path, dst)
    return dst, task, names


def _onnx_io_info(onnx_path: str):
    """Read input/output names, shapes and dynamic axes from an ONNX model.

    Returns:
        (inputs, outputs, dynamic_axes) where ``inputs``/``outputs`` map the
        tensor name to a list of dimensions ('str' entries are dynamic axis
        names), and ``dynamic_axes`` follows the mmdeploy deploy-config
        convention ``{tensor_name: {axis: axis_name}}``.
    """
    import onnx

    model = onnx.load(onnx_path)

    def dims_of(tensor):
        out = []
        for dim in tensor.type.tensor_type.shape.dim:
            if dim.dim_param:
                out.append(dim.dim_param)
            else:
                out.append(dim.dim_value)
        return out

    inputs = {t.name: dims_of(t) for t in model.graph.input}
    outputs = {t.name: dims_of(t) for t in model.graph.output}

    dynamic_axes = {}
    for name, dims in list(inputs.items()) + list(outputs.items()):
        axes = {i: d for i, d in enumerate(dims) if isinstance(d, str)}
        if axes:
            dynamic_axes[name] = axes
    return inputs, outputs, dynamic_axes


def _build_input_shapes(inputs, args):
    """Build the TensorRT optimization profiles (min/opt/max) per input.

    Assumes NCHW inputs. Batch is pinned to 1; with ``--dynamic`` the
    height/width axes get a [min_imgsz, max_imgsz] profile, otherwise the
    profile is the static shape itself.
    """
    shapes = {}
    for name, dims in inputs.items():
        if len(dims) != 4:
            raise RuntimeError(
                f'Cannot build an optimization profile for input `{name}` '
                f'with shape {dims}; only NCHW inputs are supported.')
        dyn_h, dyn_w = isinstance(dims[2], str), isinstance(dims[3], str)
        if dyn_h and dyn_w:
            channel = dims[1]
            shapes[name] = dict(
                min_shape=[1, channel, args.min_imgsz, args.min_imgsz],
                opt_shape=[1, channel, args.imgsz, args.imgsz],
                max_shape=[1, channel, args.max_imgsz, args.max_imgsz])
        else:
            shape = [
                1 if isinstance(dims[0], str) else dims[0], dims[1], dims[2],
                dims[3]
            ]
            shapes[name] = dict(
                min_shape=list(shape),
                opt_shape=list(shape),
                max_shape=list(shape))
    return shapes


def _dump_deploy_cfg(path, stem, args, inputs, outputs, dynamic_axes,
                     input_shapes):
    """Write a mmdeploy deploy config describing the conversion."""
    lines = [
        '# Auto-generated by tools/export_ultralytics.py -- do not edit by '
        'hand.', f'# Model: {stem} ({args.model})',
        f'# fp16_mode={args.fp16}, dynamic={args.dynamic}, '
        f'imgsz={args.imgsz}', '', 'onnx_config = dict(', "    type='onnx',",
        '    export_params=True,', '    keep_initializers_as_inputs=False,',
        f'    save_file={stem + ".onnx"!r},',
        f'    input_names={list(inputs)!r},',
        f'    output_names={list(outputs)!r},', '    input_shape=None,'
    ]
    if dynamic_axes:
        lines.append(f'    dynamic_axes={dynamic_axes!r},')
    lines += [
        '    optimize=True)', '', 'codebase_config = dict(',
        "    type='ultralytics',", '    model_type=\'end2end\')', '',
        'backend_config = dict(', "    type='tensorrt',",
        '    common_config=dict(', f'        fp16_mode={args.fp16},',
        f'        max_workspace_size={args.max_workspace}),',
        '    model_inputs=[dict(', f'        input_shapes={input_shapes!r})])',
        ''
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    args = parse_args()
    stem = _stem(args.model)
    work_dir = args.work_dir or osp.join('work-dirs', 'ultralytics', stem)
    work_dir = osp.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    print(f'[1/4] exporting {args.model} to ONNX ...')
    onnx_path, task, names = _export_onnx(args, work_dir)
    print(f'      onnx: {onnx_path}')

    print('[2/4] inspecting ONNX I/O ...')
    inputs, outputs, dynamic_axes = _onnx_io_info(onnx_path)
    input_shapes = _build_input_shapes(inputs, args)
    print(f'      task={task}, inputs={inputs}, outputs={list(outputs)}')

    print('[3/4] building TensorRT engine with mmdeploy ...')
    cfg_path = osp.join(work_dir, f'{stem}_tensorrt.py')
    _dump_deploy_cfg(cfg_path, stem, args, inputs, outputs, dynamic_axes,
                     input_shapes)

    from mmdeploy.apis.tensorrt import onnx2tensorrt
    engine_name = f'{stem}.engine'
    onnx2tensorrt(
        work_dir, engine_name, 0, cfg_path, onnx_path, device=args.device)
    engine_path = osp.join(work_dir, engine_name)
    print(f'      engine: {engine_path}')

    print('[4/4] writing meta ...')
    names_list = [names[i] for i in sorted(names)] \
        if isinstance(names, dict) else list(names)
    meta = dict(
        model=osp.basename(args.model),
        task=task,
        num_classes=len(names_list),
        class_names=names_list,
        imgsz=args.imgsz,
        dynamic=bool(args.dynamic),
        fp16=bool(args.fp16),
        device=args.device,
        input_names=list(inputs),
        output_names=list(outputs),
        input_shapes=input_shapes,
        dynamic_axes=dynamic_axes,
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
