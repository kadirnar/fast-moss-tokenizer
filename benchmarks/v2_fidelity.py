"""V2 stereo corpus, changed graph inputs and native streaming equivalence."""

import argparse
import gc
import json
from pathlib import Path

import torch

from benchmarks.codec_compare import source_state
from benchmarks.residual_gemv_async_model import compare
from benchmarks.v2_baseline import stereo_source
from fast_moss.graphs import GraphedCallable
from fast_moss.v2 import codec, load_model, optimized
from fast_moss.v2_loading import MODEL_ID, REVISION


def corpus(clip):
    for batch in [1, 2]:
        for frames in [1, 3, 12]:
            index = (
                torch.arange(frames * 3840, device="cuda")[None]
                + torch.arange(batch, device="cuda")[:, None] * 3840
            )
            x = clip[:, index % clip.shape[1]].permute(1, 0, 2).contiguous()
            yield f"environment_b{batch}_f{frames}", x
            if frames == 1:
                yield f"channel_swap_b{batch}", x.flip(1)
                isolated = x.clone()
                isolated[:, 1].zero_()
                yield f"left_only_b{batch}", isolated
    time = torch.arange(3840, device="cuda", dtype=torch.float32) / 48000
    tone = torch.stack(
        [
            0.2 * torch.sin(2 * torch.pi * 18000 * time),
            0.15 * torch.sin(2 * torch.pi * 21000 * time),
        ]
    )[None]
    yield "native_48k_independent_18k_21k_tones", tone
    yield "silence", torch.zeros(1, 2, 3840, device="cuda")
    impulse = torch.zeros(1, 2, 3840, device="cuda")
    impulse[0, 0, 500] = 1
    impulse[0, 1, 2900] = -1
    yield "independent_channel_impulses", impulse
    yield "tiny_stereo", tone * 1.0e-8


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/v2_fidelity.json")
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--stream-frames", type=int, default=162)
    args = p.parse_args()
    model = load_model()
    clip, source = stereo_source()
    report = {
        "scope": "v2 original eager versus optimized native BF16 compute with FP32 parameters/quantizer; true stereo and independent-channel synthetic signals",
        **source_state(),
        "model_id": MODEL_ID,
        "revision": REVISION,
        "source": source,
        "dtype_policy": model.get_codec_dtype_summary(),
        "attention_implementation": model.attention_implementation,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cases": [],
        "streaming": args.streaming,
    }

    def save():
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")

    def native(x):
        enc = model._encode_frame(x)
        return (
            enc.audio_codes,
            enc.encoder_hidden_states,
            model._decode_frame(enc.audio_codes).audio,
        )

    if not args.streaming:
        for name, x in corpus(clip):
            ref = native(x)
            changed = x.flip(1).contiguous()
            changed_ref = native(changed)
            gc.collect()
            torch.cuda.empty_cache()
            with optimized(model) as owner:
                checks = {
                    "eager": compare(ref, native(x)),
                    "fixed_adapter": compare(ref, codec(model, x)),
                }
                graph = GraphedCallable(lambda z: codec(model, z), x)
                checks["graph"] = compare(ref, graph(x))
                checks["changed_graph"] = compare(changed_ref, graph(changed))
                checks["replayed_original"] = compare(ref, graph(x))
                counts = {
                    k: getattr(owner, k)
                    for k in [
                        "linear_modules",
                        "conv_modules",
                        "quantizer_modules",
                        "cached_bytes",
                        "linear_calls",
                        "prepare_calls",
                        "rope_modules",
                        "residual_modules",
                        "rope_calls",
                        "residual_calls",
                    ]
                }
                del graph
            checks["restored"] = compare(ref, native(x))
            exact = all(c["bits_equal"] for v in checks.values() for c in v)
            report["cases"].append(
                {
                    "name": name,
                    "shape": list(x.shape),
                    "checks": checks,
                    "counts": counts,
                    "all_exact": exact,
                }
            )
            save()
            print(name, exact, counts, flush=True)
            if not exact:
                raise SystemExit("V2 corpus mismatch")
    else:
        index = (
            torch.arange(args.stream_frames * 3840, device="cuda")[None]
            + torch.arange(args.batch, device="cuda")[:, None] * 3840
        )
        x = clip[:, index % clip.shape[1]].permute(1, 0, 2).contiguous()
        # Explicit cyclic extension of the 2.48-second real stereo recording.
        enc = model.encode(x, return_dict=True, chunk_duration=0.08)
        ref = model.decode(enc.audio_codes, return_dict=True, chunk_duration=0.08)
        print("native streaming complete", args.batch, args.stream_frames, flush=True)
        with optimized(model) as owner:
            actual_enc = model.encode(x, return_dict=True, chunk_duration=0.08)
            actual_dec = model.decode(
                actual_enc.audio_codes, return_dict=True, chunk_duration=0.08
            )
            checks = {
                "encode": compare(
                    (
                        enc.audio_codes,
                        enc.audio_codes_lengths,
                        enc.encoder_hidden_states,
                    ),
                    (
                        actual_enc.audio_codes,
                        actual_enc.audio_codes_lengths,
                        actual_enc.encoder_hidden_states,
                    ),
                ),
                "decode": compare(
                    (ref.audio, ref.audio_lengths),
                    (actual_dec.audio, actual_dec.audio_lengths),
                ),
            }
            counts = {
                k: getattr(owner, k)
                for k in [
                    "cached_bytes",
                    "linear_calls",
                    "prepare_calls",
                    "select_calls",
                    "rope_calls",
                    "residual_calls",
                ]
            }
        exact = all(c["bits_equal"] for v in checks.values() for c in v)
        report.update(
            batch=args.batch,
            frames=args.stream_frames,
            seconds=args.stream_frames * 0.08,
            input_transform="both stereo channels extended cyclically; lane i offset by i*3840 samples",
        )
        report["cases"].append(
            {
                "name": "native_streaming",
                "shape": list(x.shape),
                "checks": checks,
                "counts": counts,
                "all_exact": exact,
            }
        )
        print("streaming exact", exact, counts, flush=True)
        if not exact:
            save()
            raise SystemExit("V2 streaming mismatch")
    report["all_exact"] = all(c["all_exact"] for c in report["cases"])
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    save()


if __name__ == "__main__":
    main()
