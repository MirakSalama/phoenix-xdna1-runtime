"""Exercise the installed Windows pack using Bionic's own runtime selector and chat API."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4


LMS_ROOT = Path(os.environ["USERPROFILE"]) / ".lmstudio"
CLI = LMS_ROOT / "bin/lms.exe"
PHOENIX = "phoenix-xdna1-win-x86_64-avx2@0.1.0"
CPU = "llama.cpp-win-x86_64-avx2@2.80.1"
MODELS = ["lfm2.5-230m@q4_k_m", "smollm2-360m-instruct"]
QUESTION = "What is the capital of France? Answer in a short sentence."
CSI = re.compile(r"\x1b\[[\d;?]*[A-Za-z]")
LOCAL_HTTP = build_opener(ProxyHandler({}))


def lms(*args, timeout=90):
    proc = subprocess.run([CLI, *args], capture_output=True, stdin=subprocess.DEVNULL,
                          encoding="utf-8", errors="replace", timeout=timeout,
                          creationflags=subprocess.CREATE_NO_WINDOW)
    if proc.returncode:
        raise RuntimeError(f"lms {args[0]}: {proc.stderr.strip()} {proc.stdout.strip()}")
    return CSI.sub("", proc.stdout).strip()


def inventory():
    return lms("runtime", "ls")


def selected(text):
    matches = [row.split()[0] for row in text.splitlines()
               if "✓" in row and row.split()[-1] == "GGUF" and "MLX" not in row]
    if len(matches) != 1:
        raise RuntimeError(f"Cannot identify selected GGUF engine: {text}")
    return matches[0]


def loaded_models():
    return json.loads(lms("ps", "--json"))


def server_state():
    return json.loads(lms("server", "status", "--json"))


def run(model_key, port, trace, report):
    ident = "xdna-check-" + uuid4().hex[:12]
    record = {"model": model_key}
    report["models"].append(record)
    try:
        record["load"] = lms("load", model_key, "--identifier", ident, "--context-length", "512",
                             "--parallel", "1", "--ttl", "180", "--yes", timeout=180)
        record["instance"] = next(m for m in loaded_models() if m.get("identifier") == ident)
        offset = trace.stat().st_size if trace else None
        payload = {"model": ident, "messages": [{"role": "user", "content": QUESTION}],
                   "temperature": 0, "max_tokens": 18, "stream": False}
        request = Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                          data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
        t0 = time.monotonic()
        with LOCAL_HTTP.open(request, timeout=240) as response:
            body = json.load(response)
        record.update(reply=body["choices"][0]["message"]["content"], usage=body.get("usage"),
                      elapsed_seconds=round(time.monotonic() - t0, 3))
        if trace:
            with trace.open("rb") as stream:
                stream.seek(offset)
                record["dispatch_during_request"] = stream.read().decode("utf-8", "replace").splitlines()
        print(model_key, ascii(record["reply"]), record["elapsed_seconds"], flush=True)
        if "paris" not in record["reply"].lower():
            raise RuntimeError("Expected Paris in the generated answer")
        if trace and not any("dispatch" in s.lower() for s in record["dispatch_during_request"]):
            raise RuntimeError("No hardware dispatch evidence during Bionic's reply")
    finally:
        if any(m.get("identifier") == ident for m in loaded_models()):
            lms("unload", ident)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=["cpu", "phoenix"])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, action="append")
    args = parser.parse_args()
    engine = PHOENIX if args.engine == "phoenix" else CPU
    before = inventory()
    previous = selected(before)
    if engine not in before:
        raise RuntimeError(f"Engine {engine} is absent from Bionic inventory")
    started = False
    trace = None
    report = {"requested_engine": engine, "before": before, "models": []}
    try:
        if engine != previous:
            lms("runtime", "select", engine)
        report["selected_inventory"] = inventory()
        if selected(report["selected_inventory"]) != engine:
            raise RuntimeError("Bionic did not select the requested engine")
        status = server_state()
        if not status["running"]:
            lms("server", "start", "--bind", "127.0.0.1")
            started = True
            status = server_state()
        if engine == PHOENIX:
            candidate = LMS_ROOT / "extensions/backends" / PHOENIX.replace("@","-") / "dispatch-test.log"
            candidate.touch(exist_ok=False)
            trace = candidate
        for key in args.model or MODELS:
            run(key, status["port"], trace, report)
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        try:
            if started:
                lms("server", "stop")
        finally:
            try:
                if previous != engine and selected(inventory()) == engine:
                    lms("runtime", "select", previous)
                report["restored_selection"] = selected(inventory())
                report["server_after"] = server_state()
            finally:
                if trace:
                    report["hardware_trace"] = trace.read_text(encoding="utf-8", errors="replace").splitlines()
                    trace.unlink()
                args.report.parent.mkdir(exist_ok=True, parents=True)
                args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
