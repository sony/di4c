import torch
from sdtt.models.dit import DiT
from sdtt.models.dit_orig import DIT as DiTOrig
from sdtt.models.ar_orig import AR as AROrig


def get_backbone(config, vocab_size) -> torch.nn.Module:
    # set backbone
    mtype = config.model.type
    if mtype == "ddit":
        backbone = DiT(config, vocab_size=vocab_size, adaptive=config.time_conditioning)
    elif mtype == "ddit-orig":
        backbone = DiTOrig(config, vocab_size)
    elif mtype == "ar_orig":
        backbone = AROrig(config, vocab_size + 1, vocab_size)
    else:
        raise ValueError(f"Unknown backbone: {config.backbone}")

    return backbone
