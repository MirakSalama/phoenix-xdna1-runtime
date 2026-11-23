"""Measure cold and repeated warm replies through the selected Bionic runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
from urllib.request import Request
from uuid import uuid4

from bionic_chat import (
    CPU, LMS_ROOT, LOCAL_HTTP, MODELS, QUESTION,
    inventory, lms, loaded_models, selected, server_state,
)


ORIGINAL = "phoenix-xdna1-win-x86_64-avx2@0.1.0"
COUNT = "Count from 1 to 50. Separate the numbers with commas. Start at 1 and output only the numbers."
FILLERS = ("Blue", "Rain", "Milk", "Salt", "Snow", "Gold", "Time", "Sand", "Water", "Stone", "Green", "Red")


def prefill_prompt(workload, i):
    # Distinct from the first user token onward, to prevent reuse of the timed document.
    word = FILLERS[i]
    repeats = {"prefill128": 90, "prefill160": 112, "prefill256": 218, "prefill320": 272}[workload]
    return (word + " ") * repeats + "\nWhat is the capital of France? Reply with only the city's name."


def reply(port, model_id, question, limit):
    payload = {"model": model_id, "input": question, "temperature": 0,
               "max_output_tokens": limit, "stream": True, "store": False}
    request = Request(f"http://127.0.0.1:{port}/api/v1/chat", json.dumps(payload).encode(),
                      {"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    result = None
    chunks = []
    prompt_events = []
    with LOCAL_HTTP.open(request, timeout=240) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            event = json.loads(line.partition(b":")[2])
            if event["type"] == "message.delta" and event.get("content"):
                if first is None:
                    first = time.perf_counter() - start
                chunks.append(event["content"])
            elif event["type"] == "error":
                raise RuntimeError(f"Bionic streaming failure: {event}")
            elif event["type"].startswith("prompt_processing."):
                prompt_events.append(event)
            elif event["type"] == "chat.end":
                result = event["result"]
    elapsed = time.perf_counter() - start
    if result is None or first is None:
        raise RuntimeError(f"Missing streamed result, received {len(chunks)} nonempty deltas")
    text = "".join(chunks)
    final_text = "".join(x["content"] for x in result["output"] if x["type"] == "message")
    if text != final_text or (question == QUESTION and "paris" not in text.lower()):
        raise RuntimeError(f"Unexpected reply: {text!r}, final message: {final_text!r}")
    if result["model_instance_id"] != model_id:
        raise RuntimeError(f"Unexpected instance {result['model_instance_id']!r}")
    stats = result["stats"]
    if question == COUNT and stats["total_output_tokens"] < 24:
        raise RuntimeError(f"Counting reply too short to measure sustained decoding: {text!r}")
    return {"text": text, "stats": stats, "prompt_events": prompt_events,
            "client_first_delta_seconds": first, "wall_seconds": elapsed}


def read_trace(trace, offset):
    if trace is None:
        return [], offset
    with trace.open("rb") as f:
        f.seek(offset)
        return f.read().decode("utf8", "replace").splitlines(), f.tell()


def measure_model(key, stats, trace, port, warm_runs):
    ident = "phoenix-perf-" + uuid4().hex[:10]
    entry = {"model": key, "requests": []}
    stats["models"].append(entry)
    try:
        start = time.perf_counter()
        lms("load", key, "--identifier", ident, "--context-length", "512", "--parallel", "1",
            "--ttl", "300", "--yes", timeout=200)
        entry["load_seconds"] = time.perf_counter() - start
        entry["instance"] = next(m for m in loaded_models() if m.get("identifier") == ident)
        offset = trace.stat().st_size if trace else 0
        long_prompt = stats["workload"].startswith("prefill")
        for i in range(warm_runs + 1):
            question = prefill_prompt(stats["workload"], i) if long_prompt else stats["prompt"]
            result = reply(port, ident, question, stats["max_output_tokens"])
            if long_prompt and stats["max_output_tokens"] > 1:
                if "paris" not in result["text"].lower() or result["stats"]["total_output_tokens"] >= stats["max_output_tokens"]:
                    raise RuntimeError(f"Expected an untruncated answer containing Paris, received {result['text']!r}")
            if long_prompt:
                result["prompt"] = question
                if result["stats"]["input_tokens"] < 96:
                    raise RuntimeError(f"Long prompt too short: {result['stats']}")
            result["phase"] = ("first_after_load" if i == 0 else
                               "warm_distinct_prompt" if long_prompt else "warm_repeated_prompt")
            result["hardware_trace"], offset = read_trace(trace, offset)
            entry["requests"].append(result)
        first = entry["requests"][0]
        warm = entry["requests"][1:]
        # Stock GGUF prefill and cached single-token decode can round near-tied logits differently.
        # A short known answer must match everywhere. Longer answers must match within each phase.
        comparable = entry["requests"] if stats["prompt"] == QUESTION else warm
        if not long_prompt and any(r["text"] != comparable[0]["text"] or
               r["stats"]["total_output_tokens"] != comparable[0]["stats"]["total_output_tokens"]
               for r in comparable):
            raise RuntimeError(f"Outputs diverged on repeated {key!r} requests")
        if trace and not long_prompt and not any("dispatch" in line.lower()
                             for r in entry["requests"] for line in r["hardware_trace"]):
            raise RuntimeError(f"No NPU dispatch was recorded during replies from {key!r}")
        if long_prompt and stats["prefill"] and key in stats["required_prefill_models"]:
            if not any("prefill" in line.lower() and "tensor=" in line.lower()
                       for r in entry["requests"] for line in r["hardware_trace"]):
                raise RuntimeError(f"No hardware prefill dispatch during {key!r} long-prompt run")
        median = statistics.median
        entry["warm_medians"] = {
            "wall_seconds": median(r["wall_seconds"] for r in warm),
            "first_token_seconds": median(r["stats"]["time_to_first_token_seconds"] for r in warm),
        }
        decode = [r["stats"]["tokens_per_second"] for r in warm if isinstance(r["stats"].get("tokens_per_second"), (int, float))]
        if decode:
            entry["warm_medians"]["tokens_per_second"] = median(decode)
        print(key, "cold wall", round(first["wall_seconds"], 3), "warm", entry["warm_medians"],
              "output", repr(first["text"]), flush=True)
    finally:
        if any(m.get("identifier") == ident for m in loaded_models()):
            lms("unload", ident)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", help="cpu, original, or full lms runtime alias@version")
    parser.add_argument("--warm-runs", type=int, default=4)
    parser.add_argument("--workload", choices=["paris", "count", "prefill128", "prefill160", "prefill256", "prefill320"], default="paris")
    parser.add_argument("--enable-prefill", action="store_true", help="Enable CPU/NPU prompt processing in the prefill benchmark")
    parser.add_argument("--full-answer", action="store_true", help="Generate the full city answer for each long prompt")
    parser.add_argument("--require-prefill-on", choices=MODELS, action="append", help="Require a hardware prefill trace on this model")
    parser.add_argument("--placement", choices=["all", "dense", "head", "split-head"],
                        help="Request a hardware placement while the diagnostic trace is active")
    parser.add_argument("--head-tiles", type=int, choices=range(1, 9), metavar="1..8",
                        help="Set the number of NPU tiles for split-head profiling")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Compare each reply text and token count with this CPU run")
    parser.add_argument("--allow-punctuation", action="store_true",
                        help="For count with a reference, accept punctuation differences and record them")
    args = parser.parse_args()
    if args.warm_runs < 1:
        parser.error("use at least one warm run")
    if args.report.exists():
        parser.error("choose a new report path")
    if args.allow_punctuation and (args.workload != "count" or not args.reference):
        parser.error("--allow-punctuation requires --workload count and --reference")
    engine = {"cpu": CPU, "original": ORIGINAL}.get(args.engine, args.engine)
    long_prompt = args.workload.startswith("prefill")
    if args.enable_prefill and (not long_prompt or engine == CPU):
        parser.error("--enable-prefill needs a prefill workload on a Phoenix runtime")
    if args.require_prefill_on and not args.enable_prefill:
        parser.error("--require-prefill-on requires --enable-prefill")
    if args.full_answer and not long_prompt:
        parser.error("--full-answer requires a prefill workload")
    if long_prompt and args.warm_runs >= len(FILLERS):
        parser.error("too many long-prompt runs for unique documents")
    if args.placement and engine == CPU:
        parser.error("--placement applies to a Phoenix runtime")
    if args.head_tiles is not None and args.placement != "split-head":
        parser.error("--head-tiles requires --placement split-head")
    before = inventory()
    previous = selected(before)
    if engine not in [r.split()[0] for r in before.splitlines() if r.split()]:
        raise RuntimeError(f"Bionic does not list {engine!r}")
    pack = LMS_ROOT / "extensions/backends" / engine.replace("@", "-")
    question, limit = (QUESTION, 18) if args.workload == "paris" else (COUNT, 40)
    if long_prompt:
        question, limit = args.workload, (18 if args.full_answer else 1)
    stats = {"engine": engine, "previous_engine": previous, "placement": args.placement, "head_tiles": args.head_tiles,
             "workload": args.workload, "prefill": args.enable_prefill,
             "required_prefill_models": args.require_prefill_on or ["smollm2-360m-instruct"],
             "prompt": question, "max_output_tokens": limit, "allow_punctuation": args.allow_punctuation, "models": [],
             "method": "New instance, one first request, then identical prompt repeated in new stateless API calls. "
                       "Each engine receives the same KV-cache opportunity. Parallelism 1; context 512; greedy; "
                       "stream via Bionic native API. Server reports TTFT and generation rate."}
    if long_prompt:
        stats["method"] = ("Distinct user documents from the first user token onward. First after load "
                           "reported independently. Each measured warm request has a different document. Temperature 0, context 512, parallel 1.")
    trace = None
    started = False
    try:
        if engine != previous:
            lms("runtime", "select", engine)
        current = inventory()
        if selected(current) != engine:
            raise RuntimeError("Bionic runtime selection was not applied")
        status = server_state()
        if not status["running"]:
            lms("server", "start", "--bind", "127.0.0.1")
            started = True
            status = server_state()
        if engine != CPU:
            binary = pack / "ggml-phoenix.dll"
            stats["backend_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
            file = pack / "dispatch-test.log"
            with file.open("xb") as marker:
                trace = file
                if args.placement:
                    marker.write(f"placement={args.placement}\n".encode("utf-8"))
                if args.head_tiles is not None:
                    marker.write(f"tiles={args.head_tiles}\n".encode("utf-8"))
                if args.enable_prefill:
                    marker.write(b"prefill=1\n")
        for model in MODELS:
            measure_model(model, stats, trace, status["port"], args.warm_runs)
        if args.reference:
            reference = json.loads(args.reference.read_text(encoding="utf8"))
            if reference["prompt"] != question or reference.get("max_output_tokens", 18) != limit:
                raise RuntimeError("CPU reference is from a different workload")
            refs = {m["model"]: {r.get("prompt", r["phase"]): r for r in m["requests"]} for m in reference["models"]}
            mismatches = []
            stats["reference_mismatches"] = mismatches
            for model in stats["models"]:
                for res in model["requests"]:
                    ref = refs[model["model"]][res.get("prompt", res["phase"])]
                    same_tokens = all(res["stats"][k] == ref["stats"][k] for k in ("input_tokens", "total_output_tokens"))
                    if same_tokens and res["text"] == ref["text"]:
                        continue
                    same_words = re.findall(r"[^\W_]+", ref["text"]) == re.findall(r"[^\W_]+", res["text"])
                    mismatches.append({"model": model["model"], "phase": res["phase"],
                                       "reference_text": ref["text"], "actual_text": res["text"],
                                       "same_token_counts": same_tokens, "same_words": same_words})
            stats["strict_reference_parity"] = not mismatches
            if mismatches and (not args.allow_punctuation or not all(
                m["same_token_counts"] and m["same_words"] for m in mismatches
            )):
                raise RuntimeError(f"{len(mismatches)} replies differ from the CPU reference; inspect reference_mismatches")
            stats["matches_reference"] = args.reference.name
    except BaseException as error:
        stats["error"] = repr(error)
        raise
    finally:
        try:
            if started:
                lms("server", "stop")
        finally:
            try:
                if previous != engine and selected(inventory()) == engine:
                    lms("runtime", "select", previous)
                stats["restored_selection"] = selected(inventory())
                stats["server_after"] = server_state()
            finally:
                if trace:
                    stats["trace"] = trace.read_text(encoding="utf8", errors="replace").splitlines()
                    trace.unlink()
                args.report.parent.mkdir(parents=True, exist_ok=True)
                with args.report.open("x", encoding="utf8") as output:
                    json.dump(stats, output, indent=2)


if __name__ == "__main__":
    main()
