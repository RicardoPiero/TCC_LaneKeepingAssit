"""Carga e inferencia do SegFormer-B0 para o runtime do LKA."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation


def load_checkpoint(path, device):
    """Carrega checkpoint autocontido salvo no treinamento do SegFormer."""
    checkpoint = torch.load(path, map_location=device)
    config = SegformerConfig.from_dict(checkpoint["config"])
    model = SegformerForSemanticSegmentation(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint["image_mean"], checkpoint["image_std"]


def infer_mask(model, image_bgr, eval_size, device, image_mean, image_std):
    """Retorna mascara binaria uint8 (0/255) na resolucao de avaliacao."""
    resized = cv2.resize(image_bgr, eval_size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    mean = torch.as_tensor(image_mean, dtype=tensor.dtype, device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(image_std, dtype=tensor.dtype, device=device).view(1, 3, 1, 1)
    pixel_values = (tensor - mean) / std

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = F.interpolate(
            outputs.logits,
            size=(eval_size[1], eval_size[0]),
            mode="bilinear",
            align_corners=False,
        )
        pred = logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()

    return pred * 255
