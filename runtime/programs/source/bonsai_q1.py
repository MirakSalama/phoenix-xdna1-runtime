# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Build a packed-weight NPU1 kernel for 3,072 rows of the Bonsai vocabulary head."""
import argparse
from pathlib import Path
import shutil

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1, Tile
from aie.iron.kernel import ExternalFunction
from aie.utils import config
import numpy as np

STREAMS, ROWS, COLS, GROUP = 8, 3072, 5120, 16
ROW_WORDS = (COLS // 128) * 9
PACKETS = (COLS + ROW_WORDS - 1) // ROW_WORDS
X_WORDS = PACKETS * ROW_WORDS
PROGRAM_NAME = "bonsai-q1-3072x5120"


def linear(total, offset, count):
    return TensorAccessPattern((total,), offset, [1, 1, 1, count], [0, 0, 0, 1])


@iron.jit
def bonsai_q1(weights: In, vector: In, output: Out, *, vector_scale: CompileTime[int], paired: CompileTime[int]):
    words = np.ndarray[(ROW_WORDS,), np.dtype[np.uint16]]
    activation = np.ndarray[(X_WORDS,), np.dtype[np.uint16]]
    group = np.ndarray[(GROUP,), np.dtype[np.float32]]
    source = str(Path(__file__).with_name("bonsai_q1_vec.cc" if vector_scale else "bonsai_q1.cc").resolve())
    score = ExternalFunction("bonsai_score", source_file=source, arg_types=[words, activation, group, np.int32, np.int32],
                             include_dirs=[config.cxx_header_path()])
    fanout = 2 if paired else 1
    cores = STREAMS * fanout
    in_type = np.ndarray[(fanout * ROW_WORDS,), np.dtype[np.uint16]]
    out_type = np.ndarray[(fanout * GROUP,), np.dtype[np.float32]]
    inputs = [ObjectFifo(in_type, depth=2, name=f"input{k}") for k in range(STREAMS)]
    outputs = [ObjectFifo(out_type, depth=2, name=f"scores{k}") for k in range(STREAMS)]
    if paired:
        # The eight shim DMA channels each deliver two consecutive rows to a
        # memory tile. Local DMA sends alternate rows to two compute cores.
        local_inputs = []
        local_outputs = []
        for k in range(STREAMS):
            mem = Tile(k // 2, 1)
            local_inputs.extend(inputs[k].cons(tile=mem).split([0, ROW_WORDS], tile=mem,
                obj_types=[words, words], names=[f"wi{k}a", f"wi{k}b"]))
            local_outputs.extend(outputs[k].prod(tile=mem).join([0, GROUP], tile=mem,
                obj_types=[group, group], names=[f"yo{k}a", f"yo{k}b"]))
    else:
        local_inputs, local_outputs = inputs, outputs

    def core(inp, out, x, scratch, op, copy_mode):
        for packet in range_(PACKETS):
            chunk = inp.acquire(1)
            op(chunk, x, scratch, copy_mode, packet)
            inp.release(1)
        for _ in range_(ROWS // (cores * GROUP)):
            result = out.acquire(1)
            for n in range_(GROUP):
                row = inp.acquire(1)
                op(row, x, result, 1, n)
                inp.release(1)
            out.release(1)

    workers = [Worker(core, [local_inputs[k].cons(), local_outputs[k].prod(), Buffer(activation, name=f"x{k}"),
                             Buffer(group, name=f"scratch{k}"), score, (2 + k%2 if paired else 0)],
                      tile=Tile(k // (2 * fanout), 2 + k % (2 * fanout)), dynamic_objfifo_lowering=True)
               for k in range(cores)]

    def sequence(w, x, y, sources, sinks):
        transaction = TaskGroup()
        local_rows = ROWS // STREAMS
        for k in range(STREAMS):
            x_tap = linear(X_WORDS, 0, X_WORDS)
            if paired:
                # Both paired cores eventually receive all fifteen packets:
                # the odd frame length flips their roles on the second fill.
                sources[k].fill(x, tap=x_tap, group=transaction)
                # Joined groups have 16 even rows then 16 odd rows. Scatter
                # directly into the original GGUF row order in host memory.
                out_tap = TensorAccessPattern((ROWS,), k * local_rows, [1, local_rows // (2 * GROUP), 2, GROUP],
                                              [0, 2 * GROUP, 1, 2])
            else:
                out_tap = linear(ROWS, k * local_rows, local_rows)
            sources[k].fill(x, tap=x_tap, group=transaction)
            sources[k].fill(w, tap=linear(ROWS * ROW_WORDS, k * local_rows * ROW_WORDS, local_rows * ROW_WORDS), group=transaction)
            sinks[k].drain(y, tap=out_tap, wait=True, group=transaction)
        transaction.finish()

    runtime = Runtime(sequence, [np.ndarray[(ROWS * ROW_WORDS,), np.dtype[np.uint16]], activation,
                                 np.ndarray[(ROWS,), np.dtype[np.float32]],
                                 [inputs[k].prod(tile=Tile(k // 2, 0)) for k in range(STREAMS)],
                                 [outputs[k].cons(tile=Tile(k // 2, 0)) for k in range(STREAMS)]])
    return Program(iron.get_current_device(), runtime, workers=workers).resolve_program()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--vector-scale", action="store_true")
    parser.add_argument("--paired", action="store_true")
    args = parser.parse_args()
    if args.paired and not args.vector_scale:
        parser.error("--paired requires --vector-scale")
    iron.set_current_device(NPU1())
    binary, instructions = bonsai_q1.specialize(vector_scale=int(args.vector_scale), paired=int(args.paired)).compilable.compile()
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary, args.output / (PROGRAM_NAME + ".xclbin"))
    shutil.copy2(instructions, args.output / (PROGRAM_NAME + ".bin"))
