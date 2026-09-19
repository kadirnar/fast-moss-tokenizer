import torch
from transformers import AutoModel

MODEL_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
REVISION = "3cd226ba2947efa357ef453bcad111b6eafba782"


def strict_precision():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def load_model(device="cuda"):
    strict_precision()
    return AutoModel.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True,
        dtype=torch.float32,
    ).eval().to(device).requires_grad_(False)
