import argparse
import os
import time
from collections import OrderedDict

import torch
from safetensors.torch import save_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract checkpoint["model"] and save it as safetensors.')
    parser.add_argument('checkpoint', help='Input .pt checkpoint path')
    parser.add_argument(
        '-o',
        '--output',
        default=None,
        help=('Output .safetensors path. Defaults to '
              '<checkpoint>.model.safetensors'))
    parser.add_argument(
        '--key',
        default='model',
        help=
        'Dictionary key containing the model state dict. Defaults to model.')
    parser.add_argument(
        '--raw-state-dict',
        action='store_true',
        help=('Treat the loaded checkpoint itself as a state dict when --key '
              'is absent.'))
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite output file if it already exists.')
    return parser.parse_args()


def default_output_path(checkpoint_path):
    root, _ = os.path.splitext(checkpoint_path)
    return f'{root}.model.safetensors'


def tensor_state_dict(state_dict):
    tensors = OrderedDict()
    skipped = []
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            tensors[key] = value.detach().cpu().contiguous()
        else:
            skipped.append(key)
    return tensors, skipped


def main():
    args = parse_args()
    output = args.output or default_output_path(args.checkpoint)

    if os.path.exists(output) and not args.force:
        raise FileExistsError(
            f'Output already exists: {output}. Pass --force to overwrite.')

    start = time.perf_counter()
    print(f'[convert] loading checkpoint: {args.checkpoint}', flush=True)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    print(
        f'[convert] torch.load: {time.perf_counter() - start:.1f}s',
        flush=True)

    if isinstance(checkpoint, dict) and args.key in checkpoint:
        state_dict = checkpoint[args.key]
        source = f'checkpoint[{args.key!r}]'
    elif args.raw_state_dict:
        state_dict = checkpoint
        source = 'checkpoint'
    else:
        keys = sorted(checkpoint.keys()) if isinstance(checkpoint,
                                                       dict) else []
        raise KeyError(
            f'Key {args.key!r} not found in checkpoint. Available keys: {keys[:50]}'
        )

    if not hasattr(state_dict, 'items'):
        raise TypeError(
            f'{source} is not a state dict-like object: {type(state_dict)}')

    tensors, skipped = tensor_state_dict(state_dict)
    print(
        f'[convert] tensors: {len(tensors)}, skipped non-tensors: {len(skipped)}',
        flush=True)
    if skipped:
        print(f'[convert] skipped keys sample: {skipped[:20]}', flush=True)

    metadata = {
        'source_checkpoint': os.path.abspath(args.checkpoint),
        'source_key': source,
        'format': 'pt_model_state_dict',
    }

    save_start = time.perf_counter()
    print(f'[convert] saving safetensors: {output}', flush=True)
    save_file(tensors, output, metadata=metadata)
    print(
        f'[convert] save_file: {time.perf_counter() - save_start:.1f}s',
        flush=True)
    print(f'[convert] done: {output}', flush=True)


if __name__ == '__main__':
    main()
