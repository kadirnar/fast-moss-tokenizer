"""Reconstruct a 48 kHz stereo WAV file with the optimized native codec."""

import argparse
from pathlib import Path

import soundfile as sf
import torch

from .v2 import load_model, optimized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input 48 kHz stereo WAV")
    parser.add_argument("output", type=Path, help="Output WAV, saved as FP32")
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    if args.output.suffix.lower() != ".wav":
        parser.error("Output must have a .wav extension")
    audio, sample_rate = sf.read(args.input, dtype="float32", always_2d=True)
    if sample_rate != 48_000 or audio.shape[1] != 2:
        parser.error("Input must be 48 kHz stereo (two channels)")
    if not len(audio):
        parser.error("Input audio is empty")

    model = load_model()
    waveform = torch.from_numpy(audio.T.copy()).unsqueeze(0).cuda()
    # The native encoder drops incomplete frames; pad the tail before encoding
    # and trim the reconstruction back to the original sample count below.
    padding = (-waveform.shape[-1]) % model.downsample_rate
    if padding:
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    with torch.inference_mode(), optimized(model):
        encoded = model.encode(waveform, return_dict=True, chunk_duration=0.08)
        decoded = model.decode(
            encoded.audio_codes, return_dict=True, chunk_duration=0.08
        )
        output = decoded.audio[0, :, : len(audio)].float().cpu().T.numpy()

    sf.write(args.output, output, sample_rate, subtype="FLOAT")
    print(f"Saved {args.output}: {sample_rate} Hz stereo, {len(output)} samples")


if __name__ == "__main__":
    main()
