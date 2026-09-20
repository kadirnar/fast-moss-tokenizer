"""Native v2 baseline on real stereo audio; no v1 speedup claims are reused."""

import argparse
import hashlib
import json
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF

from benchmarks.baseline import measure
from fast_moss.v2_loading import MODEL_ID, REVISION, load_model


def stereo_source():
    path = Path("data/environment.wav")
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] != 2:
        raise ValueError(
            "Expected a real stereo source; channel averaging is not allowed"
        )
    value = torch.from_numpy(audio.T.copy())
    if rate != 48000:
        value = AF.resample(value, rate, 48000)
    return value.cuda(), {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "original_sampling_rate": rate,
        "sampling_rate": 48000,
        "channels": audio.shape[1],
        "samples_per_channel": value.shape[1],
        "transform": "resample both original stereo channels to 48 kHz; no channel averaging",
    }


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/v2_baseline.json")
    p.add_argument("--frames", type=int, nargs="+", default=[1, 3])
    p.add_argument("--repeats", type=int, default=10)
    args = p.parse_args()
    model = load_model()
    clip, source = stereo_source()
    report = {
        "scope": "native 48 kHz stereo v2 eager encode/decode and direct codec roundtrip; resident input/model; excludes audio I/O and loading",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "dtype_policy": model.get_codec_dtype_summary(),
        "attention_implementation": model.attention_implementation,
        "sampling_rate": model.sampling_rate,
        "channels": model.number_channels,
        "downsample_rate": model.downsample_rate,
        "quantizers": 32,
        "source": source,
        "cases": [],
    }
    for frames in args.frames:
        x = clip[:, : frames * 3840].unsqueeze(0).contiguous()
        enc = model._encode_frame(x)
        dec = model._decode_frame(enc.audio_codes)

        def run():
            e = model._encode_frame(x)
            return (
                e.audio_codes,
                e.encoder_hidden_states,
                model._decode_frame(e.audio_codes).audio,
            )

        record = {
            "frames": frames,
            "batch": 1,
            "input_shape": list(x.shape),
            "codes_shape": list(enc.audio_codes.shape),
            "audio_shape": list(dec.audio.shape),
            "encode": measure(lambda: model._encode_frame(x), repeats=args.repeats),
            "decode": measure(
                lambda: model._decode_frame(enc.audio_codes), repeats=args.repeats
            ),
            "roundtrip": measure(run, repeats=args.repeats),
        }
        report["cases"].append(record)
        print(
            frames,
            {k: record[k]["wall_ms_median"] for k in ("encode", "decode", "roundtrip")},
            flush=True,
        )
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
