# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Build a row-streaming, eight-core Phoenix GEMV program with an FP32 result.

Each of eight independent shim DMA streams sends one control record, one
activation vector, and the worker's contiguous slice of the padded weights.
The control record holds the number of 16-row output groups on that worker.
Only the generated DMA instructions depend on row count.
"""
from pathlib import Path
import argparse
import hashlib
import shutil

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1, Tile
from aie.iron.kernel import ExternalFunction
from aie.utils import config
import numpy as np
from ml_dtypes import bfloat16

CORES = 8
GROUP = 16
COLS = 1024
HEIGHTS = (128, 256, 384, 512, 768, 1024, 1536, 2048, 2560, 3072, 4096,
           6144, 8192, 16384, 32768, 49152, 65536)

def linear(total, offset, count):
    return TensorAccessPattern((total,), offset, [1, 1, 1, count], [0, 0, 0, 1])

@iron.jit
def stream_gemv(weights: In, vector: In, output: Out, *, rows: CompileTime[int]):
    if rows % (CORES * GROUP) or not 0 < rows // (CORES * GROUP) < 65536:
        raise ValueError("rows must be a supported multiple of 128")
    chunk = np.ndarray[(COLS,), np.dtype[bfloat16]]
    group = np.ndarray[(GROUP,), np.dtype[np.float32]]
    control = np.ndarray[(1,), np.dtype[np.int32]]
    src = str(Path(__file__).with_suffix(".cc").resolve())
    op = ExternalFunction("phoenix_stream", source_file=src,
                          arg_types=[chunk, chunk, group, control, np.int32, np.int32],
                          include_dirs=[config.cxx_header_path()])
    inputs = [ObjectFifo(chunk, depth=4, name=f"weight_{k}") for k in range(CORES)]
    outputs = [ObjectFifo(group, depth=2, name=f"result_{k}") for k in range(CORES)]

    def core(inp, out, x, ctl, scratch, operation):
        header = inp.acquire(1)
        operation(header, x, scratch, ctl, 0, 0)
        inp.release(1)
        vec = inp.acquire(1)
        operation(vec, x, scratch, ctl, 1, 0)
        inp.release(1)
        for _ in range_(ctl[0]):
            result = out.acquire(1)
            for i in range_(GROUP):
                row = inp.acquire(1)
                operation(row, x, result, ctl, 2, i)
                inp.release(1)
            out.release(1)

    workers = [Worker(core, [inputs[k].cons(), outputs[k].prod(), Buffer(chunk, name=f"x{k}"),
                             Buffer(control, name=f"groups{k}"), Buffer(group, name=f"dummy{k}"), op],
                      tile=Tile(k // 2, 2 + k % 2), dynamic_objfifo_lowering=True)
               for k in range(CORES)]
    def sequence(w, x, y, prods, conss):
        g = TaskGroup()
        per_core = rows // CORES
        for k in range(CORES):
            prods[k].fill(x, tap=linear(2 * COLS, 0, 2 * COLS), group=g)
            prods[k].fill(w, tap=linear(rows * COLS, k * per_core * COLS, per_core * COLS), group=g)
            conss[k].drain(y, tap=linear(rows, k * per_core, per_core), wait=True, group=g)
        g.finish()
    runtime = Runtime(sequence, [np.ndarray[(rows * COLS,), np.dtype[bfloat16]],
                                 np.ndarray[(2 * COLS,), np.dtype[bfloat16]],
                                 np.ndarray[(rows,), np.dtype[np.float32]],
                                 [inputs[k].prod(tile=Tile(k // 2, 0)) for k in range(CORES)],
                                 [outputs[k].cons(tile=Tile(k // 2, 0)) for k in range(CORES)]])
    return Program(iron.get_current_device(), runtime, workers=workers).resolve_program()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    iron.set_current_device(NPU1())
    args.out.mkdir(parents=True, exist_ok=True)
    programs = {}
    overlay = None
    # Bootgen timestamps can change the PDI container. Compare the actual core
    # code, locks, routes, DMA endpoints and enable sequence used by the device.
    for rows in HEIGHTS:
        binary, instructions = stream_gemv.specialize(rows=rows).compilable.compile()
        cdo = Path(binary).parent / "cdo_main"
        digest = tuple(hashlib.sha256((cdo / f"main_aie_cdo_{part}.bin").read_bytes()).digest()
                       for part in ("elfs", "init", "enable"))
        if overlay is None:
            overlay = digest
        elif digest != overlay:
            raise RuntimeError(f"Hardware image differs at {rows} rows. A shared XCLBIN cannot be used.")
        if rows == 3072:
            programs["stream-gemv.xclbin"] = binary
        programs[f"stream-{rows}.bin"] = instructions
    for name, src in programs.items():
        shutil.copy2(src, args.out / name)
