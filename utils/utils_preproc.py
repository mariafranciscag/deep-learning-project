import cv2
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import ImageOps, Image
from sklearn.model_selection import GroupShuffleSplit

# ── Image Information ─────────────────────────────────────────────────────────────────
def load_image_info(path):
    with Image.open(path) as img:
        mode = img.mode
        w, h = img.size
        arr = np.array(img.convert("RGB"))
    normalized = arr / 255.0
    mean = normalized.mean()
    std  = normalized.std()
    return mode, w, h, mean, std, arr


# ── Data Split ─────────────────────────────────────────────────────────────────
def create_dl_splits(df, train_frac=0.8, val_frac=0.1, test_frac=0.1, test=True, random_state=42):
    """
    Splits the dataset into Train, Validation, and Test sets.
    Ensures all images of the same lesion_id stay in the same set to prevent leakage.
    """
    temp_frac = val_frac + test_frac
    val_relative_frac = val_frac / temp_frac

    # First Split: Train vs rest
    gss1 = GroupShuffleSplit(n_splits=1, test_size=temp_frac, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(df, groups=df['lesion_id']))

    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    if not test:
        print(f"Total images: {len(df)}")
        print(f"Train: {len(train_df)} images | {train_df['lesion_id'].nunique()} unique lesions")
        print(f"Val:   {len(temp_df)} images | {temp_df['lesion_id'].nunique()} unique lesions")

        # Leakage verification
        train_ids = set(train_df['lesion_id'])
        val_ids   = set(temp_df['lesion_id'])

        assert len(train_ids.intersection(val_ids)) == 0,  "Leakage between Train and Val!"
        print("Leakage check passed! No overlapping lesions.\n")

        return train_df, temp_df

    # Second Split: Validation vs Test
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_relative_frac, random_state=random_state)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df['lesion_id']))

    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    print(f"Total images: {len(df)}")
    print(f"Train: {len(train_df)} images | {train_df['lesion_id'].nunique()} unique lesions")
    print(f"Val:   {len(val_df)} images | {val_df['lesion_id'].nunique()} unique lesions")
    print(f"Test:  {len(test_df)} images | {test_df['lesion_id'].nunique()} unique lesions\n")

    # Leakage verification
    train_ids = set(train_df['lesion_id'])
    val_ids   = set(val_df['lesion_id'])
    test_ids  = set(test_df['lesion_id'])

    assert len(train_ids.intersection(val_ids)) == 0,  "Leakage between Train and Val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Leakage between Train and Test!"
    assert len(val_ids.intersection(test_ids)) == 0,   "Leakage between Val and Test!"
    print("Leakage check passed! No overlapping lesions.\n")

    return train_df, val_df, test_df


# ── Image Cleaning ─────────────────────────────────────────────────────────────────

def show_images(df, title, n=12):
    """Display a grid of images for visual inspection."""
    plt.figure(figsize=(15, 12))
    sample = df.sample(min(n, len(df)))

    for i, (_, row) in enumerate(sample.iterrows()):
        plt.subplot(3, 4, i+1)
        img = Image.open(row["image_path"])
        plt.imshow(img)
        plt.title(row["image_id"], fontsize=8)
        plt.axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def dull_razor(img):
    """Removes hair artifacts using morphological blackhat filtering."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    dst = cv2.inpaint(img, mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
    return dst


def fix_illumination(img):
    """Standardizes brightness via CLAHE"""
    # CLAHE on the L-channel (Lab color space)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    img_corrected = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
    return cv2.cvtColor(img_corrected, cv2.COLOR_RGB2BGR)


def remove_background(img):
    """
    Remove background from dermoscopic image using lesion segmentation.
    """
    # Convert to LAB and use L channel (better for dermoscopy)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L_channel = img_lab[:, :, 0]
    
    # Otsu thresholding (automatic, no tuning needed)
    _, mask = cv2.threshold(L_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological closing (fix: proper kernel size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Keep only largest component (remove noise)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8) * 255
    
    # Apply mask (white background)
    result = img.copy()
    result[mask == 0] = 255
    
    return result


def clean_and_save_images(df, dest_dir):
    """Apply the dull razor algorithm to a DataFrame and save results."""
    os.makedirs(dest_dir, exist_ok=True)
    df = df.copy()
    df["cleaned_path"] = ""

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Cleaning → {dest_dir}"):
        try:
            img = cv2.imread(row["image_path"])
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = dull_razor(img)

            new_filename = f"{row['image_id']}_cleaned.jpg"
            new_path = os.path.join(dest_dir, new_filename)
            cv2.imwrite(new_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            df.at[idx, "cleaned_path"] = new_path
        except Exception as e:
            print(f"Error processing {row['image_id']}: {e}")

    return df



def verify_remediation(df, threshold, num_samples=3):
    # Target images previously identified as UMAP outliers
    samples = df[df["umap_dist"] > threshold].head(num_samples)

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, num_samples * 5))

    for i, (idx, row) in enumerate(samples.iterrows()):
        # Load Original
        orig_path = row["image_path"]
        orig = cv2.imread(orig_path)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

        cleaned = cv2.imread(row["cleaned_path"])
        cleaned = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)

        axes[i, 0].imshow(orig)
        axes[i, 0].set_title(f"Original (Outlier): {row['image_id']}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(cleaned)
        axes[i, 1].set_title(f"Cleaned and Standardized - remotion of body hair")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


# ── Image Resizing ─────────────────────────────────────────────────────────────────

def format_standard_square(img):
    """1. Squishes the image to 224x224, ignoring aspect ratio."""
    return img.resize((224, 224))

def format_padded_square(img):
    """2. Scales to fit inside 224x224, padding the rest with black."""
    return ImageOps.pad(img, (224, 224), color=(0, 0, 0))

def format_center_crop(img):
    """3. Scales short edge to 224, cuts the exact 224x224 center."""
    return ImageOps.fit(img, (224, 224), centering=(0.5, 0.5))

def format_short_rectangle(img):
    """4. Hardcoded to 300x224 (maintains 4:3 ratio based on 600x450 original)."""
    return img.resize((300, 224))

def format_area_matched(img):
    """5. Hardcoded to 256x192 (maintains 4:3 ratio, matches 224x224 pixel area)."""
    return img.resize((256, 192))


# ── Final Pipeline ─────────────────────────────────────────────────────────────────
def preprocessing_pipeline(img, clahe=False, remove_background=False, resizing_strategy=format_standard_square):
    """
    Apply optional preprocessing steps to standardize dermoscopic images.
    Parameters:
    
    ----------
    img : np.ndarray
        Input image (BGR).
    clahe : bool, default=False
        Apply illumination normalization (CLAHE).
    remove_background : bool, default=False
        Remove non-diagnostic skin background.
    resizing_strategy : callable, default=format_standard_square
        Function to resize image to standard size.

    Returns:
    -------
    np.ndarray
        Preprocessed image.
    """

    if img is None:
        return None

    if clahe:
        img = fix_illumination(img)

    if remove_background:
        img = remove_background(img)

    img = resizing_strategy(img)
    img = img.astype('float32') / 255.0

    return img