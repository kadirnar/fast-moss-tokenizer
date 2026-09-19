"""Reduced random architecture for development only, never a model speed claim."""
import torch
from fast_moss.loading import strict_precision
from upstream.configuration_moss_audio_tokenizer import MossAudioTokenizerConfig
from upstream.modeling_moss_audio_tokenizer import MossAudioTokenizerModel


def structural_model():
    strict_precision()
    torch.manual_seed(42)
    config = MossAudioTokenizerConfig.from_pretrained("upstream")
    config.causal_transformer_context_duration = .16
    for group in (config.encoder_kwargs, config.decoder_kwargs):
        for block in group:
            if block["module_type"] == "Transformer":
                block["num_layers"] = 1
                block["d_model"] = 64
                block["dim_feedforward"] = 128
                block["num_heads"] = 1
    return MossAudioTokenizerModel(config).eval().cuda().requires_grad_(False)
