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
def prefill(w: In, x: In, y: Out, *, rows: CompileTime[int], columns: CompileTime[int],
            tokens: CompileTime[int], resident: CompileTime[bool]):
    return _build_design(iron.get_current_device(), rows, columns, tokens, 64, 64, tokens // 4, 4,
                         "bf16", "f32", 1, 1, False, False, False, compact=rows != 2560 and not resident)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokens", type=int, choices=(128, 256), default=128)
    parser.add_argument("--resident", action="store_true")
    args = parser.parse_args()
    iron.set_current_device(NPU1())
    args.output.mkdir(parents=True, exist_ok=True)
    for rows, columns in ((2560, 960), (7680, 2048), (6144, 1536), (12288, 1536),
                          (1536, 6144), (2048, 7680)):
        if (args.tokens > 128 or args.resident) and rows == 2560:
            continue
        if args.resident and rows == 2048:
            continue
        image, instructions = prefill.specialize(rows=rows, columns=columns, tokens=args.tokens, resident=args.resident).compilable.compile()
        family = "resident" if args.resident else "prefill"
        stem = args.output / f"{family}-{rows}x{columns}x{args.tokens}"
        shutil.copy2(image, stem.with_suffix(".xclbin"))
        shutil.copy2(instructions, stem.with_suffix(".bin"))
