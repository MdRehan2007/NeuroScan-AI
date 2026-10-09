import os
import uuid
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.segmentation import slic, mark_boundaries


def generate_shap(model, image_np, transform, class_idx, device, num_samples=80, n_segments=50):
    """
    Superpixel-based KernelSHAP explainer for image classification.
    Estimates feature attributions (Shapley values) for each superpixel.
    """
    # 1. Segment image into superpixels
    segments = slic(image_np, n_segments=n_segments, compactness=10.0, start_label=0)
    num_superpixels = int(segments.max()) + 1

    # Prepare prediction function
    def predict_batch(masked_images):
        batch = []
        for img in masked_images:
            pil_img = Image.fromarray(np.uint8(np.clip(img, 0, 255)))
            tensor = transform(pil_img)
            batch.append(tensor)
        batch_tensor = torch.stack(batch).to(device)
        with torch.no_grad():
            outputs = model(batch_tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = F.softmax(outputs, dim=1)
        return probs.cpu().numpy()[:, class_idx]

    # 2. Sample random binary mask vectors
    # Ensure all-ones and all-zeros are included for baseline calibration
    masks = np.random.randint(0, 2, size=(num_samples, num_superpixels))
    masks[0] = 1  # Full image
    masks[1] = 0  # Background image

    # Create background reference image (mean pixel value)
    bg_color = image_np.mean(axis=(0, 1))
    bg_image = np.ones_like(image_np, dtype=np.float32) * bg_color

    # 3. Build perturbed images and evaluate target class probabilities
    perturbed_images = []
    for mask in masks:
        perturbed = image_np.copy().astype(np.float32)
        for sp in range(num_superpixels):
            if mask[sp] == 0:
                perturbed[segments == sp] = bg_image[segments == sp]
        perturbed_images.append(perturbed)

    scores = predict_batch(perturbed_images)  # (N,)

    # 4. Solve KernelSHAP regression weights using Ridge regression / pseudo-inverse
    # Kernel weight: (|M| - 1) / (comb(|M|, |z'|) * |z'| * (|M| - |z'|))
    z_sums = masks.sum(axis=1)
    weights = np.ones(num_samples)
    for i in range(num_samples):
        z_len = z_sums[i]
        if z_len == 0 or z_len == num_superpixels:
            weights[i] = 100.0  # High weight for reference endpoints
        else:
            weights[i] = (num_superpixels - 1) / (z_len * (num_superpixels - z_len) + 1e-5)

    W = np.diag(np.sqrt(weights))
    X = np.dot(W, masks)
    Y = np.dot(W, scores)

    # Solve least squares for Shapley values
    shap_values, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    # 5. Map Shapley values onto 2D image map
    shap_map = np.zeros(segments.shape, dtype=np.float32)
    for sp in range(num_superpixels):
        shap_map[segments == sp] = shap_values[sp]

    # 6. Normalize and overlay on original image
    # Positive attributions highlighted red/yellow, negative blue/cyan
    abs_max = max(abs(shap_map.min()), abs(shap_map.max()), 1e-7)
    norm_shap = (shap_map / abs_max)  # [-1, +1]

    # Convert to heatmap: positive attribution -> warm colors (RED/YELLOW), negative -> cool colors
    heatmap_gray = np.uint8(127.5 * (norm_shap + 1.0))
    heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    h, w = image_np.shape[:2]
    heatmap_color = cv2.resize(heatmap_color, (w, h))

    # Add boundaries
    overlay = cv2.addWeighted(image_np, 0.6, heatmap_color, 0.4, 0)
    boundary_img = mark_boundaries(overlay / 255.0, segments, color=(1, 1, 1), mode='thin')
    shap_result = np.uint8(boundary_img * 255)
    return shap_result


def generate_shap_explanation(model, image_path: str, class_idx: int, transform, device, save_dir: str, num_samples=80) -> str:
    """
    Executes SHAP explanation on image_path for class_idx, saves output image into save_dir,
    and returns relative URL string for static serving.
    """
    os.makedirs(save_dir, exist_ok=True)

    with open(image_path, "rb") as f:
        pil_img = Image.open(f).convert("RGB")

    image_np = np.array(pil_img)

    shap_rgb = generate_shap(
        model=model,
        image_np=image_np,
        transform=transform,
        class_idx=class_idx,
        device=device,
        num_samples=num_samples
    )

    filename = f"shap_{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(save_dir, filename)

    cv2.imwrite(save_path, cv2.cvtColor(shap_rgb, cv2.COLOR_RGB2BGR))
    return f"/static/explanations/{filename}"
