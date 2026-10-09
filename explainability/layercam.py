import os
import uuid
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from explainability.gradcam import get_gradcam_target_layer, create_gradcam_image


class LayerCAM:
    """
    Layer-CAM implementation for CSCAN architecture.
    Element-wise spatial weighting combining positive gradients and feature activations.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        self.model = model
        if target_layer is None:
            self.target_layer = get_gradcam_target_layer(model)
        else:
            self.target_layer = target_layer

        self.activations = None
        self.gradients = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0]
            else:
                self.activations = output

        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self.gradients = grad_output[0]
            else:
                self.gradients = grad_output

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.handles.extend([h1, h2])

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def generate(self, image_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()

        input_tensor = image_tensor.clone().detach().requires_grad_(True)
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]

        score = output[0, class_idx]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Layer-CAM failed to capture gradients or activations.")

        gradients = self.gradients.detach()
        activations = self.activations.detach()

        # Element-wise positive weights
        pos_gradients = F.relu(gradients)
        elementwise_cam = pos_gradients * activations  # (1, C, H, W)
        cam = elementwise_cam.sum(dim=1)  # (1, H, W)

        cam = F.relu(cam)
        cam = cam[0].cpu().numpy()

        cam = cv2.resize(cam, (224, 224))
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()

        cam = np.uint8(255 * cam)
        return cam


def generate_layercam_explanation(model: nn.Module, image_path: str, class_idx: int, transform, device, save_dir: str) -> str:
    """
    Executes Layer-CAM on image_path for class_idx, saves output image into save_dir,
    and returns relative URL string for static serving.
    """
    os.makedirs(save_dir, exist_ok=True)

    with open(image_path, "rb") as f:
        pil_img = Image.open(f).convert("RGB")

    original_np = np.array(pil_img)
    tensor = transform(pil_img).unsqueeze(0).to(device)

    explainer = LayerCAM(model)
    try:
        heatmap = explainer.generate(tensor, class_idx)
        overlay_rgb = create_gradcam_image(original_np, heatmap)

        filename = f"layercam_{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(save_dir, filename)

        cv2.imwrite(save_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        return f"/static/explanations/{filename}"
    finally:
        explainer.remove_hooks()
