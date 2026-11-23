"""Create a release archive from the tested Phoenix tree using Python's standard library."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new release ZIP filename")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    lock = json.loads((root / "compatibility.json").read_text(encoding="utf8"))
    if args.output.exists():
        parser.error("output already exists")
    names = {"package.py", "README.md", "install.ps1", "build.ps1", "verify.ps1", "set-mode.ps1", "compatibility.json",
             "build/phoenix-server.exe", "build/ggml-phoenix.dll", "build/ggml-matvec-test.exe",
             "build/ggml-prefill-test.exe", "build/prefill-regression.exe", "programs/stream-gemv.xclbin",
             "programs/Apache-2.0.txt", "programs/LLVM-exception.txt",
             "vendor/ggml/LICENSE", "vendor/ggml/src/ggml-backend-impl.h", "vendor/ggml/src/ggml-common.h",
             "vendor/ggml/src/ggml-quants.h", "vendor/ggml/include/ggml.h", "vendor/ggml/include/gguf.h",
             "vendor/ggml/include/ggml-alloc.h", "vendor/ggml/include/ggml-cpu.h", "vendor/ggml/include/ggml-backend.h"}
    names.update(f"programs/prefill-{shape}.{extension}" for shape in lock["prefillPrograms"]
                 for extension in ("bin", "xclbin"))
    names.update(f"programs/resident-{shape}.{extension}" for shape in lock["residentPrograms"]
                 for extension in ("bin","xclbin"))
    for pattern in ("src/*.cpp", "src/*.h", "tests/*.cpp", "tests/*.py", "programs/stream-*.bin",
                    "programs/source/**/*.cc", "programs/source/**/*.h", "programs/source/**/*.py",
                    "programs/source/**/*.ps1", "programs/source/**/LICENSE", "THIRD_PARTY_NOTICES.md"):
        names.update(path.relative_to(root).as_posix() for path in root.glob(pattern) if path.is_file())
    names.add("THIRD_PARTY_NOTICES.md")
    names.update(p.relative_to(root).as_posix() for p in (root / "licenses").glob("*.txt"))
    entries = {name: (root / name).read_bytes() for name in sorted(names)}
    entries["SHA256SUMS.txt"] = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in entries.items()).encode("ascii")
    with ZipFile(args.output, "x") as archive:
        for name, data in entries.items():
            archive.writestr(ZipInfo(name), data, compress_type=ZIP_DEFLATED)
    with ZipFile(args.output) as archive:
        if archive.testzip() or any(archive.read(name) != data for name, data in entries.items()):
            raise RuntimeError("Release archive failed verification")
    print(f"Verified Phoenix {lock['version']} with {len(entries)} release files.")


if __name__ == "__main__":
    main()
