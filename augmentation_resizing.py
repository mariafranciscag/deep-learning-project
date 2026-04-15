import pandas as pd
import json
from torchvision import transforms
from PIL import ImageOps


# RGB WEIGHTS (IMAGE NET)
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]


# AUGMENTATION STRATEGIES

## ── STRATEGY 1 — Light (geometric only) ──────────────────────────────────────
### Safe flips and rotation, nothing that touches color or structure
strategy_1 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


## ── STRATEGY 2 — Moderate (geometry + mild color) ─────────────────────────────
### Adds subtle brightness/contrast to simulate lighting variation
strategy_2 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.05)),  # simulates hair/occlusion
])


## ── STRATEGY 3 — Aggressive (everything + elastic + perspective) ────────────
### Most aggressive — use only for extreme minority classes (vasc, df)
strategy_3 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.15, hue=0.05),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.75, 1.0)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ElasticTransform(alpha=40.0, sigma=4.0),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.35, scale=(0.02, 0.08)),
])

### Base strategy, for the majority class
strat_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


## SAVE STRATEGIES

### Save version without ToTensor and Normalize
save_strategy_1 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
])

### Save version without ToTensor and Normalize
save_strategy_2 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

### Save version without ToTensor and Normalize
save_strategy_3 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.15, hue=0.05),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.75, 1.0)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ElasticTransform(alpha=40.0, sigma=4.0),
])

save_strat_base = transforms.Compose([]) #no transformations for base when saving



# AUGMENTATION MAPS

strategy_map = {
    "nv":    strat_base,
    "mel":   strategy_1,
    "bkl":   strategy_1,
    "bcc":   strategy_2,
    "akiec": strategy_2,
    "vasc":  strategy_3,
    "df":    strategy_3,
}

save_strategy_map = {
    "nv":    save_strat_base,
    "mel":   save_strategy_1,
    "bkl":   save_strategy_1,
    "bcc":   save_strategy_2,
    "akiec": save_strategy_2,
    "vasc":  save_strategy_3,
    "df":    save_strategy_3,
}

augmentation_multiplier_map = {
    "nv":    1,
    "mel":   3,
    "bkl":   3,
    "bcc":   4,
    "akiec": 5,
    "vasc":  8,
    "df":    8,
}


# RESIZING STRATEGIES

def format_standard_square(img):
    """1. Squishes the image to 224x224, ignoring aspect ratio."""
    return img.resize((224, 224))

def format_padded_square(img):
    """2. Scales to fit inside 224x224, padding the rest with black."""
    # ImageOps.pad scales the image to fit the box without distortion, adding borders.
    return ImageOps.pad(img, (224, 224), color=(0, 0, 0))

def format_center_crop(img):
    """3. Scales short edge to 224, cuts the exact 224x224 center."""
    # ImageOps.fit scales to fill the box and trims the excess from the edges.
    return ImageOps.fit(img, (224, 224), centering=(0.5, 0.5))

def format_short_rectangle(img):
    """4. Hardcoded to 300x224 (maintains 4:3 ratio based on 600x450 original)."""
    return img.resize((300, 224))

def format_area_matched(img):
    """5. Hardcoded to 256x192 (maintains 4:3 ratio, matches 224x224 pixel area)."""
    return img.resize((256, 192))