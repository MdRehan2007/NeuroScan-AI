import os
import uuid
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """
    Automatically identifies a suitable convolutional feature layer from the
    CSCAN model architecture.
    """
    if hasattr(model, "convnext"):
        convnext = model.convnext
        if hasattr(convnext, "features"):
            # ConvNeXtTinyPretrained: torchvision Sequential
            return convnext.features[-1]
        elif hasattr(convnext, "stages"):
            # ConvNeXtTinyScratch: stages ModuleList
            return convnext.stages[-1]
        else:
            return convnext

    if hasattr(model, "dcrf") and model.dcrf is not None:
        return model.dcrf

    if hasattr(model, "awf"):
        return model.awf

    # Fallback: search for last Conv2d layer in the model
    last_conv = None
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    if last_conv is None:
        raise ValueError("Could not automatically locate a convolutional layer for Grad-CAM.")

    return last_conv


class GradCAM:
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
        """
        Generates Grad-CAM heatmap array of shape (224, 224) uint8 [0, 255].
        """
        self.model.eval()
        self.model.zero_grad()

        # Ensure image_tensor requires grad for backward pass
        input_tensor = image_tensor.clone().detach().requires_grad_(True)

        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]

        score = output[0, class_idx]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM failed to capture gradients or activations.")

        gradients = self.gradients.detach()
        activations = self.activations.detach()

        # Check activation shape: [B, C, H, W]
        if activations.ndim == 4:
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1)
        elif activations.ndim == 3: # e.g. [B, N, C]
            weights = gradients.mean(dim=1, keepdim=True)
            cam = (weights * activations).sum(dim=-1)
        else:
            raise ValueError(f"Unexpected activation shape: {activations.shape}")

        cam = F.relu(cam)
        cam = cam[0].cpu().numpy()

        cam = cv2.resize(cam, (224, 224))
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()

        cam = np.uint8(255 * cam)
        return cam


def create_gradcam_image(original_image_np: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """
    Overlays heatmaps (224x224) onto original image (RGB np array).
    Returns RGB uint8 image.
    """
    h, w = original_image_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_image_np, 0.55, heatmap_color, 0.45, 0)
    return overlay


def generate_gradcam_explanation(model: nn.Module, image_path: str, class_idx: int, transform, device, save_dir: str) -> str:
    """
    Executes Grad-CAM on image_path for class_idx, saves output image into save_dir,
    and returns the relative URL string for static serving.
    """
    os.makedirs(save_dir, exist_ok=True)

    with open(image_path, "rb") as f:
        pil_img = Image.open(f).convert("RGB")

    original_np = np.array(pil_img)
    tensor = transform(pil_img).unsqueeze(0).to(device)

    gradcam = GradCAM(model)
    try:
        heatmap = gradcam.generate(tensor, class_idx)
        overlay_rgb = create_gradcam_image(original_np, heatmap)

        filename = f"gradcam_{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(save_dir, filename)

        # Convert RGB to BGR for cv2.imwrite
        cv2.imwrite(save_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        return f"/static/explanations/{filename}"
    finally:
        gradcam.remove_hooks()