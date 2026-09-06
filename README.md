
# CSCAN Brain Tumor MRI Classification — Web App

A Flask web app that serves your existing, already-trained CSCAN model
(ConvNeXt-Tiny + Swin-T dual branch → Adaptive Weighted Fusion → Cross-Attention
Fusion → DCRF refinement → GeM-pooled classifier head). The architecture is
untouched — this app only loads it and runs inference.

## 1. Folder structure

```
CSCAN_WebApp/
│
├── app.py                     # Flask backend
├── best_cscan_model.pth       # your trained checkpoint (already included)
├── database.db                # created automatically on first run
├── requirements.txt
│
├── models/                    # your existing CSCAN architecture, unmodified
│   ├── __init__.py
│   ├── cscan.py
│   ├── convnext.py
│   ├── swin_transformer.py
│   ├── pretrained_backbones.py
│   ├── cross_attention.py
│   ├── dcrf.py
│   └── classifier.py
│
├── utils/                     # your existing config/transforms/dataset code
│   ├── __init__.py
│   ├── config.py
│   ├── transforms.py
│   └── dataset.py
│
├── templates/
│   ├── index.html
│   └── history.html
│
└── static/
    ├── css/style.css
    ├── js/main.js
    └── uploads/                # uploaded MRI images are saved here
```

Everything is already in place in this delivered folder — you don't need to
move any files around. If you ever regenerate the project from scratch, just
copy your own `models/`, `utils/`, and `best_cscan_model.pth` into this same
layout.

## 2. Install (Windows)

Open **PowerShell** or **Command Prompt** in the `CSCAN_WebApp` folder:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> **Note on first run:** because your model was trained with
> `CSCANConfig.PRETRAINED = True`, `models/cscan.py` builds its ConvNeXt/Swin
> backbones from torchvision's ImageNet weights before your checkpoint is
> loaded on top. If those weights are already cached on this machine (they
> almost certainly are, since this is the machine you trained on — cached
> under `C:\Users\<you>\.cache\torch\hub\checkpoints\`), the app starts fully
> offline. Otherwise the very first startup needs internet access to download
> them once; after that they're cached and every future startup is offline.

## 3. Run the app

```bash
python app.py
```

You should see something like:

```
[CSCAN] Model loaded successfully on device: cuda   (or cpu, if no GPU)
 * Running on http://0.0.0.0:5000
```

Then open **http://127.0.0.1:5000** in your browser.

## 4. Using the app

- **Main page (`/`)** — upload a JPG/JPEG/PNG brain MRI image (click the box
  or drag & drop), then click **Predict**. The predicted tumor class and
  confidence percentage are displayed, and the image + result are logged to
  the database automatically.
- **History page (`/history`)** — lists every past prediction (ID, filename,
  predicted class, confidence, date/time), most recent first.

## 5. How predictions work (no shortcuts, no hardcoding)

1. The uploaded file is validated (extension, then actually opened as an
   image) and saved to `static/uploads/` with a unique filename.
2. It's converted to RGB and passed through the **exact** validation/test
   transform pipeline from `utils/transforms.py`
   (`get_val_test_transforms`: CLAHE → Resize(224×224) → ToTensor →
   Normalize(mean=0.5, std=0.5)) — identical to what `test.py` used to
   produce your reported metrics.
3. The model runs in `eval()` mode under `torch.no_grad()`.
4. If `CSCANConfig.USE_TTA` is `True` (as it is by default), the prediction
   averages softmax probabilities over the image and its horizontal flip —
   the same test-time augmentation `test.py` uses.
5. The class with the highest probability and its softmax confidence are
   returned — nothing is hardcoded.
6. Class indices come directly from `CSCANConfig.CLASSES =
   ["glioma", "meningioma", "pituitary", "notumor"]`, the exact ordering
   `utils/dataset.py` used during training, so index 0 always means glioma,
   etc.

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| "Model not loaded" banner on the main page | Check the terminal for the exact error. Usually means `best_cscan_model.pth` isn't in the project root, or doesn't match the architecture in `models/cscan.py`. |
| `ModuleNotFoundError` on startup | Run `pip install -r requirements.txt` inside your activated virtual environment. |
| Predictions seem slow | Normal on CPU-only machines; a GPU (`torch.cuda.is_available()`) speeds this up automatically — no code changes needed. |
| "Unsupported file type" | Only `.jpg`, `.jpeg`, and `.png` files are accepted. |

## 7. Medical disclaimer

This system is developed for research and educational purposes and is not
intended to provide clinical diagnosis. This disclaimer is shown on every
page of the app.
