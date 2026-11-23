# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Verify the measured kernel and, optionally, the programmed contents of a fresh build."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def check(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Digest mismatch: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--programs", type=Path, default=HERE.parent)
    parser.add_argument("--rebuild-cache", type=Path,
                        help="optional fresh NPU_CACHE_HOME from build-programs.ps1 -Variant paired16")
    args = parser.parse_args()
    record = json.loads((HERE / "provenance.json").read_text(encoding="utf8"))
    for name, digest in record["sources"].items():
        check(HERE / name, digest)
    for name, digest in record["programs"].items():
        check(args.programs / name, digest)
    if args.rebuild_cache:
        # Containers can contain packaging timestamps. Compare the native core
        # code, routes/locks/initial values, start sequence and DMA instructions.
        candidates = [p.parent for p in args.rebuild_cache.glob("*/final.xclbin")]
        if not any(all((p / name).is_file() and hashlib.sha256((p / name).read_bytes()).hexdigest() == digest
                       for name, digest in record["rebuild_contents"].items()) for p in candidates):
            raise RuntimeError("The rebuild cache has no complete image with the measured programmed contents")
    print("Verified packed Bonsai paired16 kernel.")


if __name__ == "__main__":
    main()
