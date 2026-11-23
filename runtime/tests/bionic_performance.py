"""Measure cold and repeated warm replies through the selected Bionic runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
import unicodedata
from urllib.request import Request
from uuid import uuid4

from bionic_chat import (
    CPU, LMS_ROOT, LOCAL_HTTP, MODELS, QUESTION,
    inventory, lms, loaded_models, selected, server_state,
)


ORIGINAL = "phoenix-xdna1-win-x86_64-avx2@0.1.0"
COUNT = "Count from 1 to 50. Separate the numbers with commas. Start at 1 and output only the numbers."
FILLERS = ("Blue", "Rain", "Milk", "Salt", "Snow", "Gold", "Time", "Sand", "Water", "Stone", "Green", "Red")
PICKUP_CODES = ("CEDAR", "MAPLE", "LILAC", "WILLOW", "ORCHID", "PEAR", "IRIS", "PALM", "ROSE", "SAGE", "PINE", "ELM")


def document_packet(i):
    site, code = FILLERS[i], PICKUP_CODES[i]
    equipment = ("field recorder", "inspection camera", "light meter", "portable scanner")[i % 4]
    records = [
        "Receiving clerk's log. A driver brought three grey cases to the east entrance before the stores office opened. "
        "The labels listed the school workshop as the final destination, but the paper manifest still named the main depot. "
        "The clerk counted the cases, recorded the difference on the receipt, and moved them to the dry cage beside the lift. "
        "The afternoon dispatcher will ask the school which entrance can take a trolley. The cases may stay together until that "
        "call is returned. A handwritten number on one handle belongs to the carrier and cannot be used at the collection desk.",
        "Workshop service report. The repair bench tested both the mains supply and the spare battery under load. "
        "The technician found a damaged connector on the original cable, replaced it, and ran a second test with the case closed. "
        "The repaired assembly held its charge through the full check. The old cable went into the electrical recycling bin. "
        "A printed copy of the test results is in the clear sleeve inside the lid. Keep that sleeve with the equipment, "
        "since the receiving team uses it to update the service history. The repair invoice goes to accounts separately.",
        "Facilities message. Contractors will be marking the concrete outside the loading bay in the early afternoon. "
        "They do not need to enter the building, but the painted area must stay clear until the supervisor removes the cones. "
        "Delivery drivers should wait at the signed turning point and call the office before approaching the shutters. "
        "People collecting one portable item on foot can use the side entrance as usual. The receptionist has a folding trolley "
        "if somebody needs help with a heavier case. At closing time, return that trolley to the cupboard below the stairs.",
        "Dispatcher's route sheet. The van taking loan equipment to the library leaves after the lunch break. Its first "
        "stop is the archive, where the driver will collect two empty transit boxes and obtain a signature. The second stop "
        "is the branch library, where the receiving clerk will check the inventory against the yellow copy. A delay at the "
        "archive should be reported by telephone so staff at the library can keep the receiving door open. This route is "
        "for deliveries between buildings. It does not include items reserved for visitors collecting their own repaired equipment.",
        "Asset office note. One set of headphones was returned in a box bearing the name of a retired project. The asset "
        "number on the headphones matches an active loan account, so the clerk attached a temporary label and left the account "
        "open until the borrower confirms the return. No payment is due. The spare cable beside the box belongs to the training room "
        "and should go back into its drawer before the evening class. The asset office will send its regular summary on Friday. "
        "Please keep individual visitor collection receipts in the daily folder instead of attaching them to that summary.",
        "Packing team's incident note. Yesterday a parcel arrived with a cracked plastic corner although the outer "
        "carton showed no damage. The receiving clerk took photographs before removing the foam inserts and saved "
        "them under the shipment number. The carrier asked for the packaging to be held until its assessor completes "
        "the review. Store it behind the returns counter, away from the cases marked ready for collection. A replacement "
        "corner has already been ordered. The incident does not affect the separate equipment that the workshop has "
        "finished repairing, and visitors with a confirmed appointment may still collect their own cases.",
        "Training room booking. The evening instructor asked for a demonstration kit and a quiet bench with access "
        "to a mains socket. The stores team set aside a practice unit with a red label so it can be distinguished "
        "from equipment awaiting collection. The instructor should sign the training register when taking the kit and "
        "return it before the morning class. Spare batteries are in the upper drawer rather than in the carrying case. "
        "If the room is locked, leave the kit with the evening receptionist and write the name of the person who "
        "accepted it in the equipment book.",
        "Desk supervisor's handover. Compare the visitor's name with the appointment list, then bring the matching case to "
        "the counter before asking for the word on the collection slip. Do not use an invoice number or a courier's route number "
        "as a substitute. If the visitor has forgotten the word, ring the workshop and let its duty technician speak to the visitor. "
        "The signed receipt stays in the desk file. If the workshop cannot be reached, leave the item on the reserved shelf and "
        "offer another collection time. There is no need to close the repair job until the equipment has actually left the depot.",
    ]
    offset = i % len(records)
    ordered = records[offset:] + records[:offset]
    release = (f"Collection slip from {site} Depot. The {equipment} has passed its checks and is ready at the side entrance. "
               f"The pickup code for the {equipment} is {code}. The visitor should tell the clerk this exact word before "
               "signing the receipt. The copy in the service folder records the repair details, while this slip authorizes release.")
    return (f"{site} Depot dispatch packet. Read the following linked records for today's shift.\n\n"
            + "\n\n".join([*ordered, release])
            + f"\n\nWhat is the pickup code for the {equipment} at {site} Depot? Reply with just the code.")


def is_document_workload(workload):
    return workload.startswith("prefill") or workload in ("briefing", "documents1000")


def prefill_prompt(workload, i):
    # Distinct from the first user token onward, to prevent reuse of the timed document.
    word = FILLERS[i]
    if workload == "documents1000":
        return document_packet(i)
    if workload == "briefing":
        return (f"{word} Depot shift handover. The morning team checked every incoming parcel against the paper "
                "manifest before opening the loading bay. A box marked for the east annex arrived without a "
                "temperature reading and is waiting in the inspection room. The crates for the school went "
                "straight to storage because the driver came before the office opened. The afternoon dispatcher "
                "will send a separate collection notice for those crates. A visitor taking the repaired field "
                f"recorder should use the side entrance and give the clerk the pickup code {PICKUP_CODES[i]}. "
                "The recorder has a blue case and a replacement cable. Return the empty transit box to the "
                "driver after checking the contents. The older instructions in the printed binder apply only "
                "to replacement batteries. The supervisor confirmed that the evening schedule can stay as printed. "
                "What pickup code should the visitor give for the field recorder? Reply with only the code.")
    repeats = {"prefill80": 40, "prefill128": 90, "prefill160": 112, "prefill256": 218, "prefill320": 272}[workload]
    return (word + " ") * repeats + "\nWhat is the capital of France? Reply with only the city's name."


def reply(port, model_id, question, limit, reasoning_off=False):
    payload = {"model": model_id, "input": question, "temperature": 0,
               "max_output_tokens": limit, "stream": True, "store": False}
    if reasoning_off:
        payload["reasoning"] = "off"
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
        lms("load", key, "--identifier", ident, "--context-length", str(stats["context_length"]), "--parallel", "1",
            "--ttl", "300", "--yes", timeout=200)
        entry["load_seconds"] = time.perf_counter() - start
        entry["instance"] = next(m for m in loaded_models() if m.get("identifier") == ident)
        offset = trace.stat().st_size if trace else 0
        long_prompt = is_document_workload(stats["workload"])
        for i in range(warm_runs + 1):
            question = prefill_prompt(stats["workload"], i) if long_prompt else stats["prompt"]
            result = reply(port, ident, question, stats["max_output_tokens"], stats["reasoning_off"])
            if long_prompt and stats["max_output_tokens"] > 1:
                target = PICKUP_CODES[i] if stats["workload"] in ("briefing", "documents1000") else "Paris"
                correct = target.casefold() in result["text"].casefold()
                if stats["workload"] == "documents1000":
                    result["expected_code"] = target
                    correct = re.search(rf"\b{re.escape(target)}\b", result["text"]) is not None
                if not correct or result["stats"]["total_output_tokens"] >= stats["max_output_tokens"]:
                    raise RuntimeError(f"Expected an untruncated answer containing {target}, received {result['text']!r}")
            if long_prompt:
                result["prompt"] = question
                min_tokens = {"prefill80": 40, "documents1000": 800}.get(stats["workload"], 96)
                if result["stats"]["input_tokens"] < min_tokens:
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
            "first_visible_seconds": median(r["client_first_delta_seconds"] for r in warm),
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


LARGE_CITY_MODELS = {"lfm3-1.7b", "gemma-4-e2b-it"}


def city_only_answer(text):
    for segment in text.split("</think>"):
        parts = re.split(r"(Paris)", segment, flags=re.IGNORECASE | re.ASCII)
        if len(parts) != 3 or any(not c.isspace() and not unicodedata.category(c).startswith("P")
                                  for c in parts[0] + parts[2]):
            return False
    return True


def compare_reference(actual, reference, *, allow_equivalent_city_answer=False, allow_punctuation=False):
    """Compare existing raw receipts without running Bionic or modifying either input."""
    if (actual["prompt"] != reference["prompt"] or actual["workload"] != reference["workload"]
            or actual.get("max_output_tokens", 18) != reference.get("max_output_tokens", 18)
            or actual.get("context_length", 512) != reference.get("context_length", 512)
            or actual.get("reasoning_off", False) != reference.get("reasoning_off", False)):
        raise ValueError("CPU reference is from a different workload")
    if allow_equivalent_city_answer and (actual["workload"] != "prefill256"
            or actual.get("max_output_tokens", 1) <= 1 or not actual["models"]
            or any(m["model"] not in LARGE_CITY_MODELS for m in actual["models"])):
        raise ValueError("Equivalent city answers apply only to full-answer prefill256 on the tested large models")
    if allow_punctuation and actual["workload"] != "count":
        raise ValueError("Counting punctuation comparison requires the count workload")

    refs = {m["model"]: m["requests"] for m in reference["models"]}
    mismatches = []
    invalid_city = []
    compared = 0
    for model in actual["models"]:
        key = model["model"]
        if key not in refs or len(refs[key]) < len(model["requests"]):
            raise ValueError(f"Missing corresponding CPU requests for {key!r}")
        for i, res in enumerate(model["requests"]):
            ref = refs[key][i]
            if res["phase"] != ref["phase"] or res.get("prompt") != ref.get("prompt"):
                raise ValueError(f"CPU input or phase differs for {key!r} request {i + 1}")
            if allow_equivalent_city_answer and res.get("prompt") != prefill_prompt("prefill256", i):
                raise ValueError(f"Unexpected city-only prompt for {key!r} request {i + 1}")
            compared += 1
            cpu, npu = ref["stats"], res["stats"]
            same_input = cpu["input_tokens"] == npu["input_tokens"]
            same_tokens = same_input and cpu["total_output_tokens"] == npu["total_output_tokens"]
            same_raw = same_tokens and ref["text"] == res["text"]
            valid_city = allow_equivalent_city_answer and city_only_answer(ref["text"]) and city_only_answer(res["text"])
            if same_raw and (not allow_equivalent_city_answer or valid_city):
                continue
            same_words = re.findall(r"[^\W_]+", ref["text"]) == re.findall(r"[^\W_]+", res["text"])
            record = {"model": key, "phase": res["phase"], "request_number": i + 1,
                      "reference_text": ref["text"], "actual_text": res["text"],
                      "reference_input_tokens": cpu["input_tokens"], "actual_input_tokens": npu["input_tokens"],
                      "reference_output_tokens": cpu["total_output_tokens"], "actual_output_tokens": npu["total_output_tokens"],
                      "same_token_counts": same_tokens, "same_words": same_words,
                      "accepted": (valid_city and same_input) or (allow_punctuation and same_words and same_tokens)}
            if not same_raw:
                mismatches.append(record)
            if allow_equivalent_city_answer and not valid_city:
                invalid_city.append(record)
    return {"compared_replies": compared, "raw_mismatch_count": len(mismatches),
            "reference_mismatches": mismatches, "strict_reference_parity": not mismatches,
            "invalid_city_responses": invalid_city,
            "reference_comparison_passed": compared > 0 and not invalid_city and all(m["accepted"] for m in mismatches)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", help="cpu, original, or full lms runtime alias@version")
    parser.add_argument("--model", action="append", help="Installed Bionic model key; repeat to compare several")
    parser.add_argument("--reasoning-off", action="store_true", help="Disable reasoning for models that support Bionic's switch")
    parser.add_argument("--warm-runs", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512, help="Bionic model context in tokens; use 1536 or more for documents1000")
    parser.add_argument("--workload", choices=["paris", "count", "briefing", "documents1000", "prefill80", "prefill128", "prefill160", "prefill256", "prefill320"], default="paris")
    parser.add_argument("--enable-prefill", action="store_true", help="Enable CPU/NPU prompt processing in the prefill benchmark")
    parser.add_argument("--full-answer", action="store_true", help="Generate the full city answer for each long prompt")
    parser.add_argument("--require-prefill-on", action="append", help="Require a hardware prefill trace on this model")
    parser.add_argument("--placement", choices=["all", "dense", "head", "split-head"],
                        help="Request a hardware placement while the diagnostic trace is active")
    parser.add_argument("--head-tiles", type=int, choices=range(1, 33), metavar="1..32",
                        help="Set the number of NPU tiles for split-head profiling")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Compare each reply text and token count with this CPU run")
    parser.add_argument("--allow-punctuation", action="store_true",
                        help="For count with a reference, accept punctuation differences and record them")
    parser.add_argument("--allow-equivalent-city-answer", action="store_true",
                        help="For large-model prefill256, accept city-only formatting differences and record raw text")
    args = parser.parse_args()
    if args.warm_runs < 1:
        parser.error("use at least one warm run")
    if args.context_length < 1:
        parser.error("context length must be positive")
    if args.workload == "documents1000" and args.context_length < 1152:
        parser.error("documents1000 requires a longer model context; use --context-length 1536")
    if args.report.exists():
        parser.error("choose a new report path")
    if args.allow_punctuation and (args.workload != "count" or not args.reference):
        parser.error("--allow-punctuation requires --workload count and --reference")
    if args.allow_equivalent_city_answer and (not args.reference or args.workload != "prefill256"
            or not args.full_answer or not args.model or not set(args.model) <= LARGE_CITY_MODELS):
        parser.error("--allow-equivalent-city-answer requires large --model selections, --workload prefill256, --full-answer and --reference")
    engine = {"cpu": CPU, "original": ORIGINAL}.get(args.engine, args.engine)
    long_prompt = is_document_workload(args.workload)
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
        question, limit = args.workload, (18 if args.full_answer or args.workload in ("briefing", "documents1000") else 1)
    stats = {"engine": engine, "previous_engine": previous, "placement": args.placement, "head_tiles": args.head_tiles,
             "reasoning_off": args.reasoning_off, "context_length": args.context_length,
             "workload": args.workload, "prefill": args.enable_prefill,
             "required_prefill_models": args.require_prefill_on or ["smollm2-360m-instruct"],
             "prompt": question, "max_output_tokens": limit, "allow_punctuation": args.allow_punctuation,
             "allow_equivalent_city_answer": args.allow_equivalent_city_answer, "models": [],
             "method": "New instance, one first request, then identical prompt repeated in new stateless API calls. "
                       f"Each engine receives the same KV-cache opportunity. Parallelism 1; context {args.context_length}; greedy; "
                       "stream via Bionic native API. Server reports TTFT and generation rate."}
    if long_prompt:
        stats["method"] = ("Distinct user documents from the first user token onward. First after load "
                           f"reported independently. Each measured warm request has a different document. Temperature 0, context {args.context_length}, parallel 1.")
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
        for model in (args.model or MODELS):
            measure_model(model, stats, trace, status["port"], args.warm_runs)
        if args.reference:
            reference = json.loads(args.reference.read_text(encoding="utf8"))
            comparison = compare_reference(stats, reference, allow_punctuation=args.allow_punctuation,
                                           allow_equivalent_city_answer=args.allow_equivalent_city_answer)
            stats.update(comparison)
            print(f"CPU reference: {comparison['raw_mismatch_count']} raw mismatches across "
                  f"{comparison['compared_replies']} replies.", flush=True)
            if not comparison["reference_comparison_passed"]:
                raise RuntimeError("CPU reference comparison failed; inspect reference_mismatches and invalid_city_responses")
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
