import os
import sys
import torch
import torch.nn as nn
import coremltools as ct

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from engine.core import YAMLConfig


def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        state = checkpoint.get('ema', {}).get('module') if 'ema' in checkpoint else checkpoint.get('model')
        if state:
            cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model().eval()
    example_input = torch.rand(1, 3, 640, 640)
    example_size = torch.tensor([[640, 640]])
    traced = torch.jit.trace(model, (example_input, example_size))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name='images', shape=example_input.shape),
            ct.TensorType(name='orig_target_sizes', shape=example_size.shape)
        ],
        compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    mlmodel.save(args.output)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/dfine/dfine_hgnetv2_l_coco.yml', type=str)
    parser.add_argument('--resume', '-r', type=str)
    parser.add_argument('--output', '-o', default='model.mlpackage', type=str)
    args = parser.parse_args()
    main(args)
