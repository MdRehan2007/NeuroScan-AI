import os
import uuid
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from explainability.gradcam import get_gradcam_target_layer, create_gradcam_image


class ScoreCAM:
    """
    Score-CAM implementation for CSCAN architecture.
    Gradient-free class activation mapping using forward pass scores.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module = None, max_channels: int = 32):
        self.model = model
        if target_layer is None:
            self.target_layer = get_gradcam_target_layer(model)
        else:
            self.target_layer = target_layer

        self.max_channels = max_channels
        self.activations = None
        self.handle = None
        self._register_hook()

    def _register_hook(self):
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0]
            else:
                self.activations = output

        self.handle = self.target_layer.register_forward_hook(forward_hook)

    def remove_hook(self):
        if self.handle:
            self.handle.remove()
            self.handle = None

    def generate(self, image_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()

        with torch.no_grad():
            output = self.model(image_tensor)
            if isinstance(output, tuple):
                output = output[0]

            if self.activations is None:
                raise RuntimeError("Score-CAM failed to capture activations.")

            activations = self.activations[0]  # (C, H, W)
            num_channels = activations.shape[0]

            # Select top max_channels by spatial variance for efficiency
            variances = activations.var(dim=(1, 2))
            top_indices = torch.topk(variances, min(self.max_channels, num_channels)).indices

            # Baseline logit score for class_idx
            base_score = F.softmax(output, dim=1)[0, class_idx].item()

            input_size = (image_tensor.shape[2], image_tensor.shape[3])  # (224, 224)
            weights = []
            selected_maps = []

            for ch_idx in top_indices:
                act_map = activations[ch_idx : ch_idx + 1]  # (1, H, W)
                # Normalize activation map to [0, 1]
                min_v, max_v = act_map.min(), act_map.max()
                if max_v - min_v > 1e-7:
                    norm_map = (act_map - min_v) / (max_v - min_v)
                else:
                    norm_map = torch.zeros_like(act_map)

                # Upsample activation map to input size
                upsampled_map = F.interpolate(norm_map.unsqueeze(0), size=input_size, mode="bilinear", align_corners=False)
                masked_input = image_tensor * upsampled_map

                # Forward pass masked input
                masked_out = self.model(masked_input)
                if isinstance(masked_out, tuple):
                    masked_out = masked_out[0]
                prob = F.softmax(masked_out, dim=1)[0, class_idx].item()

                weights.append(prob)
                selected_maps.append(norm_map.cpu().numpy()[0])

            weights = np.array(weights)
            # Apply softmax weighting over channel scores
            weights = np.exp(weights - np.max(weights))
            weights = weights / (np.sum(weights) + 1e-7)

            cam = np.zeros(activations.shape[1:], dtype=np.float32)
            for w, map_np in zip(weights, selected_maps):
                cam += w * map_np

            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (224, 224))
            cam = cam - cam.min()
            if cam.max() != 0:
                cam = cam / cam.max()

            cam = np.uint8(255 * cam)
            return cam


def generate_scorecam_explanation(model: nn.Module, image_path: str, class_idx: int, transform, device, save_dir: str) -> str:
    """
    Executes Score-CAM on image_path for class_idx, saves output image into save_dir,
    and returns relative URL string for static serving.
    """
    os.makedirs(save_dir, exist_ok=True)

    with open(image_path, "rb") as f:
        pil_img = Image.open(f).convert("RGB")

    original_np = np.array(pil_img)
    tensor = transform(pil_img).unsqueeze(0).to(device)

    explainer = ScoreCAM(model)
    try:
        heatmap = explainer.generate(tensor, class_idx)
        overlay_rgb = create_gradcam_image(original_np, heatmap)

        filename = f"scorecam_{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(save_dir, filename)

        cv2.imwrite(save_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        return f"/static/explanations/{filename}"
    finally:
        explainer.remove_hook()
