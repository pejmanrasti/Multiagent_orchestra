"""
workspace_files.py — the files the FRAMEWORK owns and writes into the workspace.

CRITICAL DESIGN RULES
  1. The agent may only edit train.py. It must NEVER touch data.py or the
     evaluation script (score.py writes eval_model.py into the workspace).
  2. Three splits, three purposes:
       train      -> fitting the model                       (train.py)
       validation -> model selection + error analysis       (every iteration)
       test       -> ONE evaluation of the final model       (after the loop)
     data.py only hands out train + validation. The test accessors refuse to run
     unless the framework sets PIPELINE_ALLOW_TEST=1, which it does exactly once,
     for the final evaluation of the selected model.

Point a loader at your own dataset by changing the `_load()` block (tabular) or
PATH (images); the rest of the system is unchanged.
"""

TEST_FLAG = "PIPELINE_ALLOW_TEST"   # env var that unlocks the test split (framework only)
SEED_ENV = "PIPELINE_SEED"          # env var carrying the experiment seed into train.py

# Tokens generated code may not contain: they are the only ways to reach test data.
FORBIDDEN_IN_TRAIN = ("get_test_split", "get_test_loader", TEST_FLAG, "predictions_test")


# =========================================================================== #
# TABULAR MODE  (scikit-learn style arrays)
# `__LOAD__` is replaced by a block defining `_load() -> (X, y, class_names)`.
# =========================================================================== #
TABULAR_DATA_TEMPLATE = '''\
import os
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42                 # fixed split seed (independent of the experiment seed)
TEST_FRACTION = 0.2
VAL_FRACTION = 0.2        # fraction of the non-test data used for validation
_TEST_FLAG = "PIPELINE_ALLOW_TEST"

__LOAD__

def _split(seed=SEED):
    X, y, names = _load()
    y = np.asarray(y)
    ids = np.arange(len(y))
    rest, te = train_test_split(ids, test_size=TEST_FRACTION, random_state=seed, stratify=y)
    tr, va = train_test_split(rest, test_size=VAL_FRACTION, random_state=seed, stratify=y[rest])
    return X, y, names, tr, va, te

def _check_test_access():
    if os.environ.get(_TEST_FLAG) != "1":
        raise PermissionError("The test split is locked during optimisation; "
                              "only the framework's final evaluation may read it.")

def get_splits(seed=SEED):
    """Training + validation data ONLY: X_train, y_train, X_val, y_val."""
    X, y, _, tr, va, _ = _split(seed)
    return X[tr], y[tr], X[va], y[va]

def get_test_split(seed=SEED):
    """Framework-only. Raises unless the final evaluation unlocked it."""
    _check_test_access()
    X, y, _, _, _, te = _split(seed)
    return X[te], y[te]

def get_class_names():
    return [str(n) for n in _load()[2]]

def get_sample_ids(split, seed=SEED):
    """Stable identifiers (row indices) for the samples of a split, in loader order."""
    if split == "test":
        _check_test_access()
    _, _, _, tr, va, te = _split(seed)
    return [int(i) for i in {"train": tr, "val": va, "test": te}[split]]
'''

_DIGITS_LOAD = '''\
from sklearn.datasets import load_digits

def _load():
    X, y = load_digits(return_X_y=True)
    return X, y, [str(i) for i in range(10)]
'''


def tabular_data_py(load_block: str) -> str:
    return TABULAR_DATA_TEMPLATE.replace("__LOAD__", load_block.strip())


# Built-in demo dataset (sklearn digits, 10 classes).
DATA_PY = tabular_data_py(_DIGITS_LOAD)


# The CONTRACT the agent's train.py must satisfy (given to the coder agent as rules).
TRAIN_CONTRACT = """\
Your script MUST:
  1. `from data import get_splits` and call it as
     `X_train, y_train, X_val, y_val = get_splits()`. There is NO test data available
     to you; never try to obtain it.
  2. Read the seed with `SEED = int(os.environ.get("PIPELINE_SEED", "42"))` and use it
     for every source of randomness (random_state=SEED, np.random.seed(SEED), ...).
  3. Train a scikit-learn compatible model (or Pipeline) on the training data only.
     Use the validation data only for monitoring / early stopping.
  4. Save it with `import joblib; joblib.dump(model, "model.pkl")`. The model must
     implement predict(); prefer models that also implement predict_proba() (wrap with
     CalibratedClassifierCV if needed) so validation loss can be measured.
  5. If training is iterative, write `history.json`: a list of per-epoch rows
     {"epoch", "train_loss", "train_acc", "val_loss", "val_acc"} (omit unknown keys).
  6. Print exactly one final JSON line: {"train_acc": <float>, "val_acc": <float>}.
Do NOT modify data.py, do NOT read evaluation files. Output ONLY the Python code.
"""


# =========================================================================== #
# IMAGE / DEEP-LEARNING MODE
# The loader exposes PyTorch DataLoaders + num_classes. It only resizes images to a
# common size (so they can be batched) and yields float tensors in [0, 1]; it does
# NOT normalise. Model-specific preprocessing (exact input size, channels,
# normalisation) lives in preprocess.py, inside the saved model, so training,
# validation and the final test always feed the model the same way.
# =========================================================================== #
IMAGE_DATA_PY = '''\
import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PATH = __PATH__
IMG_SIZE = 160          # default loading size; pass image_size=model.input_spec["image_size"]
BATCH = 8               # fits a small (~2 GB) GPU like a 940MX
SEED = 42               # fixed split seed (independent of the experiment seed)
TEST_FRACTION = 0.2
VAL_FRACTION = 0.16
_TEST_FLAG = "PIPELINE_ALLOW_TEST"

def _tf(augment, image_size):
    aug = [transforms.RandomHorizontalFlip()] if augment else []
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        *aug,
        transforms.ToTensor(),      # float in [0, 1]; normalisation is done by preprocess.py
    ])

def _indices(seed=SEED):
    # Same permutation torch.utils.data.random_split would produce.
    n = len(datasets.ImageFolder(PATH))
    n_test = max(1, int(TEST_FRACTION * n)); n_val = max(1, int(VAL_FRACTION * n))
    n_train = n - n_val - n_test
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]

def _check_test_access():
    if os.environ.get(_TEST_FLAG) != "1":
        raise PermissionError("The test split is locked during optimisation; "
                              "only the framework's final evaluation may read it.")

def _loader(idx, augment, shuffle, batch_size, image_size):
    ds = Subset(datasets.ImageFolder(PATH, transform=_tf(augment, int(image_size))), idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

def get_loaders(seed=SEED, batch_size=BATCH, augment=True, image_size=IMG_SIZE):
    """Training + validation ONLY: (train_loader, val_loader, num_classes)."""
    tr, va, _ = _indices(seed)
    return (_loader(tr, augment, True, batch_size, image_size),
            _loader(va, False, False, batch_size, image_size),
            len(get_class_names()))

def get_test_loader(seed=SEED, batch_size=BATCH, image_size=IMG_SIZE):
    """Framework-only. Raises unless the final evaluation unlocked it."""
    _check_test_access()
    return _loader(_indices(seed)[2], False, False, batch_size, image_size)

def get_class_names():
    return datasets.ImageFolder(PATH).classes

def get_sample_ids(split, seed=SEED):
    """File paths of a split's samples, in (unshuffled) loader order."""
    if split == "test":
        _check_test_access()
    samples = datasets.ImageFolder(PATH).samples
    idx = dict(zip(("train", "val", "test"), _indices(seed)))[split]
    return [samples[i][0] for i in idx]
'''


def image_data_py(path: str) -> str:
    return IMAGE_DATA_PY.replace("__PATH__", repr(path))


# Framework-owned input adapter, written into the workspace in image mode. Saved
# models contain AdaptiveInput, so this file must sit next to model.pt to load it.
PREPROCESS_PY = '''\
"""
preprocess.py — framework-owned input adapter for image models (do not edit).

prepare_model(model, weights=... | image_size=...) wraps a network in AdaptiveInput,
which, on every forward pass, VERIFIES the incoming batch and ADAPTS it:
  - shape:    (C,H,W) -> (1,C,H,W); anything else is rejected with a clear error
  - range:    0-255 inputs are rescaled to [0, 1]; already-normalised inputs are
              rejected (normalising twice silently ruins accuracy)
  - channels: grayscale -> RGB, RGBA -> RGB
  - size:     resized to the model's required input size if it differs
  - normalisation with the model's own mean/std
The required size / mean / std come from the pretrained weights' transforms when
`weights` is given (e.g. ViT_B_16 -> 224), otherwise from the explicit arguments.
Because the adapter is part of the saved model, validation and the final test
evaluation preprocess exactly like training did.

check_model(model, loader, num_classes) runs one batch through the model before
training, so size/shape/class-count problems fail fast with a readable message.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_spec(weights=None, image_size=None, mean=None, std=None):
    spec = {}
    if weights is not None:
        t = weights.transforms()
        crop = t.crop_size[0] if isinstance(t.crop_size, (list, tuple)) else t.crop_size
        spec = {"image_size": int(crop), "mean": list(t.mean), "std": list(t.std)}
    if image_size is not None:
        spec["image_size"] = int(image_size)
    if "image_size" not in spec:
        raise ValueError("prepare_model needs `weights=` (pretrained) or `image_size=`.")
    spec["mean"] = list(mean if mean is not None else spec.get("mean", IMAGENET_MEAN))
    spec["std"] = list(std if std is not None else spec.get("std", IMAGENET_STD))
    if len(spec["mean"]) != len(spec["std"]):
        raise ValueError("mean and std must have the same number of channels.")
    return spec


class AdaptiveInput(nn.Module):
    def __init__(self, model, spec):
        super().__init__()
        self.model = model
        self.input_spec = dict(spec)
        c = len(spec["mean"])
        self.register_buffer("mean", torch.tensor(spec["mean"], dtype=torch.float32).view(1, c, 1, 1))
        self.register_buffer("std", torch.tensor(spec["std"], dtype=torch.float32).view(1, c, 1, 1))

    def adapt(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError(f"expected an image batch (B, C, H, W), got shape {tuple(x.shape)}")
        x = x.float()
        lo, hi = float(x.min()), float(x.max())
        if hi > 1.5:                      # 0-255 pixel values
            x, lo = x / 255.0, lo / 255.0
        if lo < -0.01:
            raise ValueError("inputs are already normalised (negative values); remove "
                             "transforms.Normalize - AdaptiveInput normalises itself.")
        c = self.mean.shape[1]
        if x.shape[1] != c:
            if x.shape[1] == 1:
                x = x.expand(-1, c, -1, -1)
            elif x.shape[1] == 4 and c == 3:
                x = x[:, :3]
            else:
                raise ValueError(f"model expects {c} channels, got {x.shape[1]}")
        size = self.input_spec["image_size"]
        if tuple(x.shape[-2:]) != (size, size):
            try:
                x = F.interpolate(x, size=(size, size), mode="bilinear",
                                  align_corners=False, antialias=True)
            except (RuntimeError, NotImplementedError):   # antialias unsupported on device
                x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, x):
        return self.model(self.adapt(x))


def prepare_model(model, weights=None, image_size=None, mean=None, std=None):
    """Wrap `model` so every input is verified, resized and normalised for it."""
    if isinstance(model, AdaptiveInput):
        return model
    return AdaptiveInput(model, make_spec(weights, image_size, mean, std))


def check_model(model, loader, num_classes, device="cpu"):
    """Run one batch through the model; raise a readable error if anything mismatches."""
    if not isinstance(model, AdaptiveInput):
        raise TypeError("wrap the network with preprocess.prepare_model(...) before training.")
    x, _ = next(iter(loader))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(x[:2].to(device))
    except Exception as e:
        raise RuntimeError(f"model failed on a sample batch of shape {tuple(x.shape)} "
                           f"(adapted to {model.input_spec['image_size']}px): {e}") from e
    finally:
        model.train(was_training)
    if out.dim() != 2 or out.shape[1] != num_classes:
        raise ValueError(f"model output shape {tuple(out.shape)}; expected (batch, {num_classes}). "
                         f"Replace the final layer for {num_classes} classes.")
    print(f"input check ok: data {tuple(x.shape[1:])} -> model {model.input_spec['image_size']}px, "
          f"{num_classes} outputs", flush=True)
'''


# Contract the agent's image train.py must satisfy.
TRAIN_CONTRACT_IMAGE = """\
You are writing a PyTorch image classifier for a SMALL GPU (~2 GB VRAM). Your script MUST:
  1. Read the seed with `SEED = int(os.environ.get("PIPELINE_SEED", "42"))` and call
     torch.manual_seed(SEED) and random.seed(SEED) before building the model.
  2. Use the GPU when available (CUDA, or Apple MPS), else CPU.
  3. Build the network for `num_classes = len(get_class_names())` (from data). Prefer
     LIGHTWEIGHT torchvision backbones with transfer learning to fit limited VRAM
     (mobilenet_v3_small, resnet18, efficientnet_b0 ...), final layer replaced for
     num_classes, unless the instructions say otherwise. Do NOT define a custom nn.Module
     class (the evaluator cannot import it); compose torchvision/nn modules instead.
  4. INPUT PREPROCESSING IS HANDLED BY THE FRAMEWORK. Wrap the network:
       from preprocess import prepare_model, check_model
       model = prepare_model(net, weights=W)        # pretrained: size/mean/std from W
       model = prepare_model(net, image_size=IMG_SIZE)   # from scratch (IMG_SIZE from data)
     then load data at the model's input size and verify before training:
       train_loader, val_loader, num_classes = get_loaders(image_size=model.input_spec["image_size"])
       model.to(device); check_model(model, train_loader, num_classes, device)
     get_loaders(seed=42, batch_size=8, augment=True, image_size=160) yields tensors in
     [0, 1]: NEVER add transforms.Normalize (the wrapper normalises). Do NOT pass unknown
     keyword arguments. There is NO test data available to you; never try to obtain it.
     Extra augmentation may be added to train_loader.dataset.dataset.transform (before ToTensor).
  5. After EACH epoch, print and flush a line like
       Epoch 1/3 - loss: 0.83 - train_acc: 0.72 - val_acc: 0.70
     and, at the end, write `history.json`: a list of per-epoch rows
     {"epoch", "train_loss", "train_acc", "val_loss", "val_acc"}.
  6. Keep memory small: do NOT increase batch size beyond 8.
  7. Save the FULL WRAPPED model: `torch.save(model, "model.pt")` (the AdaptiveInput, not
     the inner network). If you use early stopping, save the best validation epoch.
  8. Print exactly one final JSON line: {"train_acc": <float>, "val_acc": <float>}.
Do NOT modify data.py, do NOT read evaluation files. Output ONLY the Python code.
"""
