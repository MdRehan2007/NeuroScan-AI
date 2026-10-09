import os
import uuid
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries


def generate_lime(model, image_np, transform, class_idx, device, num_samples=100):
    """
    Generates a LIME superpixel visualization for the target class_idx.
    image_np: RGB numpy array of shape (H, W, 3).
    """
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
        batch = []
        for img in images:
            img_uint8 = np.uint8(np.clip(img, 0, 255))
            pil_img = Image.fromarray(img_uint8)
            tensor = transform(pil_img)
            batch.append(tensor)

        batch_tensor = torch.stack(batch).to(device)

        with torch.no_grad():
            outputs = model(batch_tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probabilities = F.softmax(outputs, dim=1)

        return probabilities.cpu().numpy()

    explanation = explainer.explain_instance(
        image_np,
        predict_fn,
        top_labels=4,
        hide_color=0,
        num_samples=num_samples
    )

    temp, mask = explanation.get_image_and_mask(
        class_idx,
        positive_only=False,
        num_features=8,
        hide_rest=False
    )

    lime_result = mark_boundaries(temp / 255.0, mask)
    lime_result = np.uint8(lime_result * 255)
    return lime_result


def generate_lime_explanation(model, image_path: str, class_idx: int, transform, device, save_dir: str, num_samples=100) -> str:
    """
    Executes LIME explanation on image_path for class_idx, saves output image into save_dir,
    and returns relative URL string for static serving.
    """
    os.makedirs(save_dir, exist_ok=True)

    with open(image_path, "rb") as f:
        pil_img = Image.open(f).convert("RGB")

    image_np = np.array(pil_img)

    lime_rgb = generate_lime(
        model=model,
        image_np=image_np,
        transform=transform,
        class_idx=class_idx,
        device=device,
        num_samples=num_samples
    )

    filename = f"lime_{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(save_dir, filename)

    cv2.imwrite(save_path, cv2.cvtColor(lime_rgb, cv2.COLOR_RGB2BGR))
    return f"/static/explanations/{filename}"