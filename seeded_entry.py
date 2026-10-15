"""Seed the existing entry point without changing shared training code."""
import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

name = sys.argv.pop(1)
assert name in ('main_synthetic', 'finetuning')
module = importlib.import_module(name)
parser = argparse.ArgumentParser(parents=[module.get_args_parser()])
args = parser.parse_args()
out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# Custom CUDA attention may remain nondeterministic; do not claim bitwise reproducibility.
(out / 'seed_control.json').write_text(json.dumps({
    'seed': args.seed, 'python': True, 'numpy': True, 'torch_cpu_cuda': True,
    'cudnn_benchmark': False, 'cudnn_deterministic': True,
    'bitwise_determinism_guaranteed': False,
}, indent=2))
module.main(args)
