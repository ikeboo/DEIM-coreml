from typing import Optional, Tuple, Union

import torch

try:
    import coremltools as ct
except ImportError as e:
    raise ImportError('Please install coremltools to use export_coreml.') from e


def export_coreml(model: Optional[Union[torch.nn.Module, torch.jit.ScriptModule]] = None,
                  model_path: Optional[str] = None,
                  output_path: str = 'model.mlpackage',
                  input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
                  convert_to: str = 'mlprogram',
                  compute_unit: str = 'cpu',
                  compute_precision: str = 'float32',
                  deployment_target: Optional[str] = None,
                  **kwargs):
    """Export a PyTorch model to CoreML ``mlpackage``.

    Parameters
    ----------
    model : torch.nn.Module or torch.jit.ScriptModule, optional
        Model instance to convert. If ``None``, ``model_path`` must be provided.
    model_path : str, optional
        Path to a serialized PyTorch model. Ignored if ``model`` is provided.
    output_path : str, default "model.mlpackage"
        Destination path of the exported CoreML package.
    input_shape : tuple, default ``(1, 3, 640, 640)``
        Example input shape used for tracing when ``model`` is not a ``ScriptModule``.
    convert_to : {"mlprogram", "neuralnetwork"}, default "mlprogram"
        Target CoreML representation.
    compute_unit : {"cpu", "gpu", "all"}, default "cpu"
        Compute unit for the CoreML model.
    compute_precision : {"float32", "float16", "fp16"}, default "float32"
        Precision of weights in the exported model.
    deployment_target : str, optional
        Minimum deployment target such as ``"iOS17"`` or ``"macOS14"``.
    **kwargs : dict
        Additional arguments forwarded to ``coremltools.convert``.
    """
    if model is None:
        if model_path is None:
            raise ValueError('Either model or model_path must be provided.')
        model = torch.load(model_path, map_location='cpu')

    model.eval()

    if not isinstance(model, torch.jit.ScriptModule):
        example_input = torch.rand(*input_shape)
        model = torch.jit.trace(model, example_input)
    else:
        example_input = torch.rand(*input_shape)

    cu_map = {
        'cpu': ct.ComputeUnit.CPU_ONLY,
        'gpu': ct.ComputeUnit.CPU_AND_GPU,
        'all': ct.ComputeUnit.ALL,
    }
    cu = cu_map.get(compute_unit.lower(), ct.ComputeUnit.ALL)

    precision_map = {
        'float32': ct.precision.FLOAT32,
        'fp16': ct.precision.FLOAT16,
        'float16': ct.precision.FLOAT16,
    }
    precision = precision_map.get(compute_precision.lower(), ct.precision.FLOAT32)

    min_target = None
    if deployment_target:
        min_target = getattr(ct.target, deployment_target, None)

    inputs = [ct.TensorType(shape=input_shape)]
    mlmodel = ct.convert(
        model,
        inputs=inputs,
        compute_units=cu,
        convert_to=convert_to,
        compute_precision=precision,
        minimum_deployment_target=min_target,
        **kwargs,
    )

    mlmodel.save(output_path)
    print(f'Saved CoreML model to {output_path}')
    return mlmodel


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Export PyTorch model to CoreML package.')
    parser.add_argument('-m', '--model-path', type=str, help='Path to torch model file.')
    parser.add_argument('-o', '--output', type=str, default='model.mlpackage', help='Output CoreML package.')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size of example input.')
    parser.add_argument('--input-size', type=int, nargs=2, default=(640, 640), help='Input height and width.')
    parser.add_argument('--convert-to', type=str, default='mlprogram', choices=['mlprogram', 'neuralnetwork'])
    parser.add_argument('--compute-unit', type=str, default='cpu', choices=['cpu', 'gpu', 'all'])
    parser.add_argument('--compute-precision', type=str, default='float32', choices=['float32', 'float16', 'fp16'])
    parser.add_argument('--deployment-target', type=str, help='Minimum deployment target (e.g. iOS17).')
    args = parser.parse_args()

    export_coreml(
        model_path=args.model_path,
        output_path=args.output,
        input_shape=(args.batch_size, 3, args.input_size[0], args.input_size[1]),
        convert_to=args.convert_to,
        compute_unit=args.compute_unit,
        compute_precision=args.compute_precision,
        deployment_target=args.deployment_target,
    )
