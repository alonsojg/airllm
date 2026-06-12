import argparse
import csv
import gc
import json
import re
import statistics
import time
from contextlib import redirect_stdout
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import psutil
import torch
from accelerate.utils.modeling import set_module_tensor_to_device

from airllm import AutoModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark AirLLM single-layer vs multi-layer loading behavior."
    )
    parser.add_argument(
        "--model",
        default="garage-bAInd/Platypus2-7B",
        help="HF repo id or local model path.",
    )
    parser.add_argument(
        "--compression",
        default="4bit",
        choices=["4bit", "8bit", "none"],
        help="Compression mode to use.",
    )
    parser.add_argument(
        "--layer-options",
        default="1,2,4,8",
        help="Comma-separated requested max_layers_in_memory values.",
    )
    parser.add_argument(
        "--prompt",
        default="Write a concise summary of why memory-aware layer loading improves LLM inference stability.",
        help="Prompt text used for generation runs.",
    )
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer max_length.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Generated tokens per run.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup generation runs per setting.")
    parser.add_argument("--runs", type=int, default=3, help="Measured generation runs per setting.")
    parser.add_argument(
        "--output-dir",
        default="air_llm/tests/benchmark_reports",
        help="Directory where CSV/JSON reports are written.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional HF token for private/gated models.",
    )
    parser.add_argument(
        "--disable-runtime-pos-emb",
        action="store_true",
        help=(
            "Benchmark workaround: disable runtime position_embeddings injection to avoid "
            "known rotary shape mismatch on some transformers/airllm combinations."
        ),
    )
    return parser.parse_args()


def parse_layer_options(layer_options):
    values = []
    for item in layer_options.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 1:
            raise ValueError("layer option values must be >= 1")
        values.append(value)
    if not values:
        raise ValueError("no valid layer option values were provided")
    return values


def current_memory_snapshot(device):
    vm = psutil.virtual_memory()
    snapshot = {
        "ram_used_gb": (vm.total - vm.available) / (1024 ** 3),
        "ram_available_gb": vm.available / (1024 ** 3),
    }

    if device.startswith("cuda") and torch.cuda.is_available():
        idx = torch.device(device).index or 0
        free_b, total_b = torch.cuda.mem_get_info(idx)
        snapshot.update(
            {
                "vram_free_gb": free_b / (1024 ** 3),
                "vram_total_gb": total_b / (1024 ** 3),
                "vram_allocated_gb": torch.cuda.memory_allocated(idx) / (1024 ** 3),
                "vram_reserved_gb": torch.cuda.memory_reserved(idx) / (1024 ** 3),
                "vram_peak_allocated_gb": torch.cuda.max_memory_allocated(idx) / (1024 ** 3),
            }
        )
    else:
        snapshot.update(
            {
                "vram_free_gb": None,
                "vram_total_gb": None,
                "vram_allocated_gb": None,
                "vram_reserved_gb": None,
                "vram_peak_allocated_gb": None,
            }
        )

    return snapshot


def cleanup_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_load_only_pass(model, requested_chunk_size):
    """
    Benchmark layer loading/moving/eviction only (no forward compute).
    This is a robust fallback when generation is incompatible in the runtime.
    """
    layer_items = list(zip(model.layer_names, model.layers))
    chunk_start = 0

    while chunk_start < len(layer_items):
        safe_chunk_size = model.detect_max_layers_in_memory(
            safety_factor=model.memory_safety_factor,
            verbose=False,
        )
        chunk_size = max(1, min(requested_chunk_size, safe_chunk_size))
        chunk = layer_items[chunk_start:chunk_start + chunk_size]
        chunk_moved = []

        for layer_name, _layer in chunk:
            state_dict = model.load_layer_to_cpu(layer_name)
            moved_layers = model.move_layer_to_device(state_dict)
            chunk_moved.append(moved_layers)

        for (_layer_name, layer), moved_layers in zip(chunk, chunk_moved):
            if model.hf_quantizer is not None:
                for param_name in moved_layers:
                    set_module_tensor_to_device(model.model, param_name, "meta")
            else:
                layer.to("meta")

            layer.to("meta")

        chunk_start += len(chunk)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_safe_limit_from_warning(stdout_text):
    match = re.search(r"safe limit=(\d+)", stdout_text)
    if not match:
        return None
    return int(match.group(1))


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def run_single_setting(args, requested_layers, device):
    compression = None if args.compression == "none" else args.compression
    print(f"\n=== Benchmark setting: requested max_layers_in_memory={requested_layers} ===")

    init_stdout = StringIO()
    with redirect_stdout(init_stdout):
        model = AutoModel.from_pretrained(
            args.model,
            compression=compression,
            max_layers_in_memory=requested_layers,
            hf_token=args.hf_token,
        )

    if args.disable_runtime_pos_emb:
        model.get_runtime_pos_emb_args = lambda seq, position_ids_args: {}
    init_output = init_stdout.getvalue()

    effective_layers_init = int(getattr(model, "max_layers_in_memory", requested_layers))
    detected_safe_limit = int(model.detect_max_layers_in_memory(verbose=False))
    init_clamp_warned = "WARNING: requested max_layers_in_memory=" in init_output
    init_safe_limit_from_warning = extract_safe_limit_from_warning(init_output)

    print(
        f"Requested={requested_layers}, effective(after init clamp)={effective_layers_init}, "
        f"safe_limit_now={detected_safe_limit}, init_warning={init_clamp_warned}"
    )

    tokens = model.tokenizer(
        [args.prompt],
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=args.max_length,
    )
    input_ids = tokens["input_ids"]
    if device.startswith("cuda") and torch.cuda.is_available():
        input_ids = input_ids.cuda()

    benchmark_mode = "generation"
    generation_error = None

    try:
        for _ in range(args.warmup_runs):
            _ = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                use_cache=False,
                return_dict_in_generate=True,
            )
    except Exception as ex:
        benchmark_mode = "load_only"
        generation_error = f"{type(ex).__name__}: {ex}"
        print(
            "Generation path failed in this environment; "
            "falling back to load-only benchmark mode. "
            f"error={generation_error}"
        )

    run_rows = []
    for run_idx in range(args.runs):
        if device.startswith("cuda") and torch.cuda.is_available():
            idx = torch.device(device).index or 0
            torch.cuda.reset_peak_memory_stats(idx)

        before = current_memory_snapshot(device)
        run_stdout = StringIO()

        start = time.perf_counter()
        with redirect_stdout(run_stdout):
            if benchmark_mode == "generation":
                generation_output = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=False,
                    return_dict_in_generate=True,
                )
                seq = generation_output.sequences[0]
                generated_tokens = int(max(0, seq.shape[0] - input_ids.shape[1]))
            else:
                run_load_only_pass(model, requested_chunk_size=effective_layers_init)
                generated_tokens = 0

        elapsed_s = time.perf_counter() - start
        tokens_per_sec = (generated_tokens / elapsed_s) if elapsed_s > 0 else 0.0

        runtime_output = run_stdout.getvalue()
        runtime_clamp_warned = "runtime memory pressure reduced safe layer chunk size" in runtime_output

        after = current_memory_snapshot(device)

        row = {
            "requested_layers": requested_layers,
            "effective_layers_init": effective_layers_init,
            "detected_safe_limit": detected_safe_limit,
            "init_clamp_warned": init_clamp_warned,
            "runtime_clamp_warned": runtime_clamp_warned,
            "run_index": run_idx,
            "benchmark_mode": benchmark_mode,
            "generation_error": generation_error,
            "elapsed_s": elapsed_s,
            "generated_tokens": generated_tokens,
            "tokens_per_s": tokens_per_sec,
            "ram_used_gb_before": before["ram_used_gb"],
            "ram_used_gb_after": after["ram_used_gb"],
            "ram_available_gb_after": after["ram_available_gb"],
            "vram_peak_allocated_gb": after["vram_peak_allocated_gb"],
            "vram_reserved_gb_after": after["vram_reserved_gb"],
            "vram_free_gb_after": after["vram_free_gb"],
            "init_safe_limit_from_warning": init_safe_limit_from_warning,
        }
        run_rows.append(row)

        print(
            f"run={run_idx + 1}/{args.runs} elapsed={elapsed_s:.3f}s "
            f"generated={generated_tokens} tok/s={tokens_per_sec:.2f} "
            f"runtime_clamp_warning={runtime_clamp_warned}"
        )

    elapsed_summary = summarize([r["elapsed_s"] for r in run_rows])
    tps_summary = summarize([r["tokens_per_s"] for r in run_rows])

    setting_summary = {
        "requested_layers": requested_layers,
        "effective_layers_init": effective_layers_init,
        "detected_safe_limit": detected_safe_limit,
        "benchmark_mode": benchmark_mode,
        "generation_error": generation_error,
        "init_clamp_warned": init_clamp_warned,
        "init_safe_limit_from_warning": init_safe_limit_from_warning,
        "runtime_clamp_warned_any": any(r["runtime_clamp_warned"] for r in run_rows),
        "elapsed_s_mean": elapsed_summary["mean"],
        "elapsed_s_stdev": elapsed_summary["stdev"],
        "tokens_per_s_mean": tps_summary["mean"],
        "tokens_per_s_stdev": tps_summary["stdev"],
        "tokens_per_s_min": tps_summary["min"],
        "tokens_per_s_max": tps_summary["max"],
    }

    cleanup_model(model)
    return setting_summary, run_rows


def attach_speedup_vs_baseline(setting_summaries, baseline_requested=1):
    baseline = None
    for s in setting_summaries:
        if s["requested_layers"] == baseline_requested:
            baseline = s
            break

    if baseline is None or baseline["tokens_per_s_mean"] <= 0:
        for s in setting_summaries:
            s["speedup_vs_baseline_x"] = None
        return

    base_tps = baseline["tokens_per_s_mean"]
    for s in setting_summaries:
        s["speedup_vs_baseline_x"] = s["tokens_per_s_mean"] / base_tps


def write_reports(output_dir, full_report, setting_summaries, run_rows):
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "benchmark_layer_loading_report.json"
    csv_settings_path = output_dir / "benchmark_layer_loading_settings.csv"
    csv_runs_path = output_dir / "benchmark_layer_loading_runs.csv"

    json_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")

    if setting_summaries:
        with csv_settings_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(setting_summaries[0].keys()))
            writer.writeheader()
            writer.writerows(setting_summaries)

    if run_rows:
        with csv_runs_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
            writer.writeheader()
            writer.writerows(run_rows)

    return json_path, csv_settings_path, csv_runs_path


def main():
    args = parse_args()
    layer_options = parse_layer_options(args.layer_options)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    started_at = datetime.now(UTC).isoformat()

    print("Starting benchmark")
    print(f"device={device} model={args.model} compression={args.compression}")
    print(f"layer_options={layer_options} runs={args.runs} warmup_runs={args.warmup_runs}")

    setting_summaries = []
    run_rows = []
    for requested_layers in layer_options:
        setting_summary, rows = run_single_setting(args, requested_layers, device)
        setting_summaries.append(setting_summary)
        run_rows.extend(rows)

    attach_speedup_vs_baseline(setting_summaries, baseline_requested=1)

    ended_at = datetime.now(UTC).isoformat()
    full_report = {
        "meta": {
            "started_at": started_at,
            "ended_at": ended_at,
            "device": device,
            "model": args.model,
            "compression": args.compression,
            "layer_options": layer_options,
            "max_new_tokens": args.max_new_tokens,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "prompt": args.prompt,
        },
        "settings": setting_summaries,
        "runs": run_rows,
    }

    output_dir = Path(args.output_dir)
    json_path, csv_settings_path, csv_runs_path = write_reports(
        output_dir, full_report, setting_summaries, run_rows
    )

    print("\n=== Summary (settings) ===")
    for s in setting_summaries:
        speedup = s["speedup_vs_baseline_x"]
        speedup_txt = f"{speedup:.2f}x" if speedup is not None else "N/A"
        print(
            f"requested={s['requested_layers']} effective={s['effective_layers_init']} "
            f"safe={s['detected_safe_limit']} tps_mean={s['tokens_per_s_mean']:.2f} "
            f"speedup_vs_1={speedup_txt} init_warn={s['init_clamp_warned']} "
            f"runtime_warn_any={s['runtime_clamp_warned_any']}"
        )

    print("\nReport files:")
    print(str(json_path))
    print(str(csv_settings_path))
    print(str(csv_runs_path))


if __name__ == "__main__":
    main()