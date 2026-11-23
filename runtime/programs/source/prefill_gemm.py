# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compile Phoenix's 16-core, BF16-input, FP32-output batched projection."""
from pathlib import Path
import argparse
import shutil
import sys

import aie.iron as iron
from aie.iron import CompileTime, In, Out
from aie.iron.device import NPU1

sys.path.insert(0, str(Path(__file__).resolve().parent / "amd-whole-array"))
from whole_array import _build_design


@iron.jit
def prefill(w: In, x: In, y: Out, *, rows: CompileTime[int], columns: CompileTime[int]):
    return _build_design(iron.get_current_device(), rows, columns, 128, 64, 64, 32, 4,
                         "bf16", "f32", 1, 1, False, False, False, compact=rows != 2560)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    iron.set_current_device(NPU1())
    args.output.mkdir(parents=True, exist_ok=True)
    for rows, columns in ((2560, 960), (7680, 2048), (6144, 1536), (12288, 1536),
                          (1536, 6144), (2048, 7680)):
        image, instructions = prefill.specialize(rows=rows, columns=columns).compilable.compile()
        stem = args.output / f"prefill-{rows}x{columns}x128"
        shutil.copy2(image, stem.with_suffix(".xclbin"))
        shutil.copy2(instructions, stem.with_suffix(".bin"))
