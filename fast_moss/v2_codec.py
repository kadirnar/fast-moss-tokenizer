"""Fixed-frame offline encode/decode adapters for native stereo v2."""

import sys

import torch


@torch.no_grad()
def encode_fixed(model, audio):
    """Graph-safe full-frame v2 encoding with unchanged tensor arithmetic.

    Only equal-length complete offline stereo frames are accepted. Their valid
    code length is known from shape, eliminating upstream's CUDA `.item()` sync.
    Use upstream encode/batch_encode for tails or variable per-lane lengths.
    """
    if (
        audio.ndim != 3
        or audio.shape[1] != 2
        or audio.shape[0] < 1
        or audio.shape[-1] < 3840
        or audio.shape[-1] % 3840
        or model.sampling_rate != 48000
        or model.number_channels != 2
        or model.attention_implementation != "sdpa"
    ):
        raise ValueError(
            "Expected complete equal-length B,2,T 48 kHz stereo frames with SDPA"
        )
    if any(getattr(m, "is_streaming", False) for m in model.modules()):
        raise RuntimeError(
            "Fixed v2 graphs are offline; use upstream streaming methods for stateful input"
        )
    lengths = torch.full(
        (audio.shape[0],), audio.shape[-1], device=audio.device, dtype=torch.long
    )
    hidden, lengths = model._flatten_channels_for_codec(audio, lengths)
    with model._codec_inference_autocast():
        for module in model.encoder:
            hidden, lengths = module(hidden, lengths)
    _, codes, code_lengths = model.quantizer(hidden.float(), lengths, None)
    output_type = sys.modules[type(model).__module__].MossAudioTokenizerEncoderOutput
    return output_type(
        audio_codes=codes,
        audio_codes_lengths=code_lengths,
        encoder_hidden_states=hidden.float(),
    )


def codec(model, audio):
    """Return owned-by-caller codes, hidden states and stereo reconstruction."""
    enc = encode_fixed(model, audio)
    return (
        enc.audio_codes,
        enc.encoder_hidden_states,
        model._decode_frame(enc.audio_codes).audio,
    )
