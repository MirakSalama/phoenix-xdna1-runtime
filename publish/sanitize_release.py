"""Build public Phoenix release snapshots from a verified local pack. Python 3.9+."""
import argparse
import hashlib
import json
import re
import stat
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


VERSION = r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
PACK = "phoenix-xdna1-win-x86_64-avx2"
SCRIPTS = {"build.ps1", "install.ps1", "verify.ps1", "set-mode.ps1", "package.py"}
NATIVE = {"build/phoenix-server.exe", "build/ggml-phoenix.dll", "build/ggml-matvec-test.exe",
          "build/ggml-prefill-test.exe", "build/prefill-regression.exe"}
NOTICES = """# Third-party notices

The GGML headers under `vendor/ggml` come from llama.cpp. They retain their MIT license in `vendor/ggml/LICENSE`. The pinned GGML revision for this release is {rev}.

Phoenix AI's initial first-generation XDNA chat work grew from Abhiram Jayasankar’s Ryzen AI NPU Lab. Its MIT notice is included in `licenses/ryzen-ai-npu-lab-MIT.txt`. [Upstream experiment archive](https://github.com/AbhiramJayasankar/ryzenai-npu-lab/tree/a9531b87b770fd3142164b61254fb6f74fed2d22)

The device program sources identify their licenses in their file headers. Device programs adapted from AMD MLIR-AIE use Apache License 2.0 with the LLVM exception. Copies of both texts accompany the programs in `programs/Apache-2.0.txt` and `programs/LLVM-exception.txt`. Bundled whole-array source, where present, retains its upstream notices and license in `programs/source/amd-whole-array/LICENSE`. Phoenix parameterizes those kernels for its own operation shapes and packing layouts.

The device kernels use AMD’s header-only AIE API. Its MIT notice is included in `licenses/aie-api-MIT.txt`. [AMD AIE API](https://github.com/Xilinx/aie_api)

The Windows host links to AMD XRT userspace interfaces under Apache License 2.0. XRT headers used to build the host carry copyrights of Xilinx, Inc. and Advanced Micro Devices, Inc. The SDK's upstream license text is included in `licenses/xrt-sdk-LICENSE.txt`. XRT and its driver are provided by the user's installation. Bionic's native engine and bindings are also obtained from the user's installation. The archive contains the Phoenix host binaries and device programs built for this version.

Copyright and license markings apply to the components that carry them. This package does not grant a separate project-wide license for otherwise unlicensed material.
"""
PRIVATE = re.compile(
    r"(?:[A-Za-z]:[\\/](?:Users|Dev|Documents and Settings)[\\/]|/(?:Users|home|root)/[^\s/]+/|"
    r"/mnt/[A-Za-z]/(?:Users|Dev)/|-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----|"
    r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{35,}|sk-[A-Za-z0-9_-]{24,}|"
    r"hf_[A-Za-z0-9]{25,}|AKIA[0-9A-Z]{16})\b)", re.I)
SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}
    return (SAFE_NAME.fullmatch(name) is not None and
            all(p not in (".", "..") and not p.startswith(".") and not p.endswith(".") and
                p.split(".", 1)[0].upper() not in reserved for p in name.split("/")))


def read_verified(source):
    with ZipFile(source) as archive:
        encoded = {}
        seen = set()
        total = 0
        for entry in archive.infolist():
            name = entry.filename
            if not safe_name(name) or name.casefold() in seen or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("Invalid path or duplicate ZIP member")
            seen.add(name.casefold())
            total += entry.file_size
            if entry.file_size > 128 << 20 or total > 512 << 20 or len(seen) > 10000:
                raise ValueError("Archive exceeds the bounds of this runtime pack format")
            encoded[name] = archive.read(entry)
    records = encoded.pop("SHA256SUMS.txt").decode("ascii").splitlines()
    signed = {}
    for line in records:
        match = re.fullmatch(r"([a-f0-9]{64})  (.+)", line)
        if not match or match[2] in signed or not safe_name(match[2]):
            raise ValueError("Malformed original checksum manifest")
        signed[match[2]] = match[1]
    if signed != {name: digest(data) for name, data in encoded.items()}:
        raise ValueError("Archive contents do not match the original release manifest")
    return encoded


def payload_type(name):
    if name in NATIVE:
        return "native"
    if re.fullmatch(r"programs/(?:gemv-3072x1024|stream-(?:gemv|[0-9]+)|(?:prefill|resident)-[0-9]+x[0-9]+x[0-9]+)\.(?:bin|xclbin)", name):
        return "device"
    if (name in SCRIPTS | {"compatibility.json", "vendor/ggml/LICENSE",
                           "programs/Apache-2.0.txt", "programs/LLVM-exception.txt"} or
            re.fullmatch(r"(?:src/[\w.-]+\.(?:cpp|h)|tests/[\w.-]+\.(?:cpp|py)|vendor/ggml/(?:include|src)/[\w.-]+\.h)", name) or
            re.fullmatch(r"programs/source/(?:[\w.-]+/)*(?:[\w.-]+\.(?:py|ps1|cc|h)|LICENSE)", name)):
        return "source"
    raise ValueError(f"Unrecognized publishable file: {name}")


def screen(name, content, binary):
    if binary:
        texts = [content.decode("latin1"), content.decode("utf-16le", "ignore"),
                 content[1:].decode("utf-16le", "ignore")]
    else:
        texts = [content.decode("utf-8-sig")]
    if any(PRIVATE.search(t) for t in texts):
        raise ValueError(f"Private path or credential pattern in {name}")


def portable_source(name, data):
    if name not in {"build.ps1", "verify.ps1", "programs/source/build-programs.ps1", "package.py"}:
        return data
    s = data.decode("utf-8-sig").replace("\r\n", "\n")
    if name == "build.ps1":
        s = re.sub(r"(\[string\]\$XrtSdk\s*=\s*)'[a-zA-Z]:[^'\n]+'", r"\g<1>$env:XRT_ROOT", s)
        s = s.replace("$root = $PSScriptRoot",
                      "if (!$XrtSdk -and !$LauncherOnly) { throw 'Supply -XrtSdk or set XRT_ROOT to the XRT SDK directory.' }\n$root = $PSScriptRoot", 1)
    elif name == "verify.ps1":
        s = re.sub(r"(\[string\]\$ModelRoot\s*=\s*)'[a-zA-Z]:[^'\n]+'",
                   r"\g<1>(Join-Path $env:USERPROFILE '.lmstudio/models')", s)
    elif name.endswith("/build-programs.ps1"):
        s = re.sub(r"(\[string\]\$Toolchain\s*=\s*)'[a-zA-Z]:[^'\n]+'", r"\g<1>$env:PHOENIX_IRON_ROOT", s)
        s = s.replace("$pack = (Resolve-Path", "if (!$Toolchain) {throw 'Supply -Toolchain or set PHOENIX_IRON_ROOT to the IRON toolchain directory.'}\n$pack = (Resolve-Path", 1)
        s = s.replace("C:/Windows/System32/AMD", "$env:SystemRoot/System32/AMD")
    elif '    entries = {' in s:
        s = s.replace('    entries = {', '    names.add("THIRD_PARTY_NOTICES.md")\n    names.update(p.relative_to(root).as_posix() for p in (root / "licenses").glob("*.txt"))\n    entries = {', 1)
        s = s.replace('"results/*"', '"THIRD_PARTY_NOTICES.md"')
    return s.encode("utf8")


def readme(version, base, raw, notes):
    command = f"lms runtime select {PACK}@{version}"
    checks = ("Run `.\\verify.ps1 -ModelRoot <model-library>` with the installed test GGUFs. "
              "The verifier opens local Bionic instances and compares CPU and Phoenix responses. " +
              ("Add `-IncludeLargeModels` for the LFM3 and Gemma test suite. " if b"IncludeLargeModels" in raw.get("verify.ps1", b"") else "") +
              ("The known LFM3 city fixture accepts equivalent text with `-AllowEquivalentCityAnswer`. " if b"AllowEquivalentCityAnswer" in raw.get("verify.ps1", b"") else "")
              if "verify.ps1" in raw else
              "Use the local Bionic chat test in `tests/bionic_chat.py` to compare CPU and Phoenix on the installed test GGUFs. Supply a fresh filename using `--report`.")
    about = notes.strip() if notes else "This release provides a versioned XDNA 1 GGUF accelerator for Bionic. Model coverage and performance depend on the hardware, model, prompt and runtime combination."
    return f"""# Phoenix XDNA 1 for Bionic {version}

{about}

## Install

Use Windows x64, a Phoenix or Hawk Point XDNA 1 device with a working AMD XRT driver, PowerShell 7 and the official Bionic CPU llama.cpp {base} AVX2 runtime. Install that CPU runtime in Bionic first. Keep GGUF models in your normal library.

Extract the ZIP to a new directory. In PowerShell 7, run the following from that directory.

```powershell
.\\install.ps1
{command}
```

Alternatively select **Phoenix XDNA 1 (Windows) {version}** in Bionic's GGUF runtime settings. Unload and reload the model. To roll back, select the previous engine and reload the model.

## Verify and build

{checks}

The release archive includes the Phoenix host and compiled NPU programs. To rebuild the host, install Visual Studio C++ Build Tools, the Windows SDK and a matching AMD XRT SDK. Run `.\\build.ps1 -XrtSdk <sdk-directory>`. The third-party NPU toolchain is separate. The device sources and their attribution are under `programs`. Where included, `programs/source/build-programs.ps1 -Toolchain <toolchain-directory>` rebuilds the device programs with a compatible IRON toolchain layout.

The SHA-256 manifest covers every release file except the manifest itself. Review `THIRD_PARTY_NOTICES.md` and the accompanying license texts.
""".encode("utf8")


def sanitize(source, root, notes_path):
    match = re.fullmatch(rf"Phoenix-XDNA1-({VERSION})\.zip", source.name)
    if not match:
        raise ValueError("Expected a Phoenix-XDNA1-X.Y.Z.zip archive")
    version = match[1]
    raw = read_verified(source)
    if "compatibility.json" in raw:
        lock = json.loads(raw["compatibility.json"])
        if lock["version"] != version or lock["pack"] != PACK or lock["basePack"] != "llama.cpp-win-x86_64-avx2":
            raise ValueError("Incompatible release identity")
        base, revision = lock["baseVersion"], lock["llamaRevision"]
        if not re.fullmatch(VERSION, base) or not re.fullmatch(r"[0-9a-f]{7,40}", revision):
            raise ValueError("Unexpected base version or source revision")
    else:
        if version != "0.1.0" or b"$version = '0.1.0'" not in raw["install.ps1"]:
            raise ValueError("Missing compatibility manifest")
        base, revision = "2.80.1", "29b52d398"
    contents = {}
    for name, data in raw.items():
        if name == "README.md" or name.startswith("results/"):
            continue
        kind = payload_type(name)
        if kind == "native" and not data.startswith(b"MZ"):
            raise ValueError(f"Invalid Phoenix host program: {name}")
        if kind == "device" and name.endswith(".xclbin") and not data.startswith(b"xclbin2"):
            raise ValueError(f"Invalid device image: {name}")
        if kind == "source":
            data = portable_source(name, data)
        screen(name, data, kind != "source")
        contents[name] = data
    required = {"build/phoenix-server.exe", "build/ggml-phoenix.dll", "src/ggml-phoenix.cpp",
                "install.ps1", "build.ps1", "vendor/ggml/LICENSE", "programs/Apache-2.0.txt",
                "programs/LLVM-exception.txt"}
    if not required <= contents.keys():
        raise ValueError("Missing required runtime files or license notices")
    default_notes = Path(__file__).with_name("notes") / (version + ".md")
    notes = (notes_path.read_text(encoding="utf8") if notes_path else
             default_notes.read_text(encoding="utf8") if default_notes.exists() else "")
    contents["README.md"] = readme(version, base, raw, notes)
    contents["THIRD_PARTY_NOTICES.md"] = NOTICES.format(rev=revision).encode("utf8")
    for notice in ("aie-api-MIT.txt", "ryzen-ai-npu-lab-MIT.txt", "xrt-sdk-LICENSE.txt"):
        data = (Path(__file__).with_name("notices") / notice).read_bytes()
        screen(notice, data, False)
        contents["licenses/" + notice] = data
    for name in ("README.md","THIRD_PARTY_NOTICES.md"):
        screen(name, contents[name], False)
    contents["SHA256SUMS.txt"] = "".join(f"{digest(contents[name])}  {name}\n" for name in sorted(contents)).encode("ascii")
    variants = root / "versions"
    assets = root / "assets"
    variants.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    destination, public = variants / version, assets / source.name
    checksum = assets / (public.name + ".sha256")
    if any(p.exists() for p in (destination, public, checksum)):
        raise ValueError("Public outputs already exist. Use a fresh staging root.")
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        temp = Path(tmp)
        archive, snapshot = temp / public.name, temp / version
        with ZipFile(archive, "x") as output:
            for name in sorted(contents):
                path = snapshot / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents[name])
                info = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                output.writestr(info, contents[name], compress_type=ZIP_DEFLATED, compresslevel=9)
        if read_verified(archive) != {k: v for k, v in contents.items() if k != "SHA256SUMS.txt"}:
            raise ValueError("Public ZIP failed roundtrip verification")
        out_sha = digest(archive.read_bytes())
        sha_file = temp / checksum.name
        sha_file.write_bytes(f"{out_sha}  {public.name}\n".encode("ascii"))
        snapshot.rename(destination)
        archive.rename(public)
        sha_file.rename(checksum)
    print(f"{version}: {len(contents)} verified files, {public.stat().st_size} bytes, SHA256 {out_sha}")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--notes", type=Path, help="Reviewed public release notes")
    args = cli.parse_args()
    sanitize(args.archive, args.output, args.notes)
