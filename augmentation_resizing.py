import pandas as pd
import json
from torchvision import transforms
from PIL import Image, ImageOps
import os
import uuid
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

with open("label2idx.json", "r") as f:
    label2idx = json.load(f)

## path for the new images
base_path = "./data" 
aug_dir = os.path.join(base_path, "HAM10000_augmented")
if not os.path.exists(aug_dir):
    os.makedirs(aug_dir)


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


# AUGMENTATION PIPELINE

def augment_single_image(original_path, transform, new_id, resize_strategy):
    """
    Load, augment, and save a single image.
    
    Args:
        original_path: Path to original image
        transform: Augmentation transform function
        new_id: New image ID
    
    Returns:
        Path to saved augmented image
    """
    img = Image.open(original_path).convert('RGB')
    img = resize_strategy(img)
    
    if transform:
        img = transform(img)
    
    new_filename = f"{new_id}.jpg"
    save_path = os.path.join(aug_dir, new_filename)
    img.save(save_path, "JPEG",)
    
    return save_path


def build_metadata_row(new_id, save_path, label, lesion_id, original_row):
    """
    Build metadata row for augmented image.
    
    Args:
        new_id: New image ID
        save_path: Path to saved image
        label: Class label
        lesion_id: New lesion ID
        original_row: Original image metadata
    
    Returns:
        Dictionary with augmented image metadata
    """
    new_row_dict = {
        'image_id': new_id,
        'image_path': save_path,
        'dx': label,
        'dataset': 'augmented',
        'lesion_id': lesion_id
    }
    
    # Copy other metadata from original row
    for col in ['dx_type', 'age', 'sex', 'localization']:
        if col in original_row:
            new_row_dict[col] = original_row[col]
    
    return new_row_dict


def offline_augmentation(df, label, multiplier, resize_strategy):
    """
    Generate augmented images for a specific class label.
    
    Optimizations:
    - Pre-compute metadata rows as dicts (O(1) lookup)
    - Get transform once, not per iteration
    - Pre-generate all UUIDs in batch
    - Single Image.open() per augmentation
    
    Args:
        df: Training dataframe
        label: Class label to augment
        multiplier: Multiplication factor (e.g., 2 = double the class)
    
    Returns:
        List of metadata dictionaries for augmented images
    """
    new_rows = []
    
    # Filter to class data
    class_df = df[df['dx'] == label]
    paths = class_df['image_path'].tolist()
    metadata_rows = class_df.to_dict('records')
    
    # Calculate augmentations needed
    n_to_generate = int(len(paths) * (multiplier - 1))
    transform = save_strategy_map.get(label)
    
    # Pre-generate all UUIDs (2 per augmented image: image_id + lesion_id)
    uuids = [uuid.uuid4().hex[:8] for _ in range(n_to_generate * 2)]
    
    for i in range(n_to_generate):
        idx = i % len(paths)
        original_path = paths[idx]
        original_row = metadata_rows[idx]
        
        # Augment image
        new_id = f"AUG_{label}_{uuids[i*2]}"
        save_path = augment_single_image(original_path, transform, new_id, resize_strategy)
        
        # Build metadata
        lesion_id = f"AUG_LESION_{label}_{uuids[i*2+1]}"
        new_row = build_metadata_row(new_id, save_path, label, lesion_id, original_row)
        new_rows.append(new_row)
    
    return new_rows


def generate_augmented_dataset(df, aug_map, resize_strategy):
    """
    Generate augmented images in parallel for all classes.
    
    Args:
        df: Training dataframe
        aug_map: {label: multiplier} mapping
    
    Returns:
        DataFrame with augmented image metadata
    """
    all_new_metadata = []
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        
        # Submit augmentation tasks
        for label, multiplier in aug_map.items():
            if multiplier > 1:
                print(f"Submitting task for augmenting {label} (multiplier: {multiplier})")
                futures.append(
                    executor.submit(offline_augmentation, df, label, multiplier, resize_strategy)
                )
        
        # Collect results
        for future in tqdm(futures, desc="Generating all augmented images"):
            all_new_metadata.extend(future.result())
    
    return pd.DataFrame(all_new_metadata)


def random_undersample_class(df, label, target_size, random_state=42):
    """
    Undersample a specific class to target size.
    
    Args:
        df: Training dataframe
        label: Class label to undersample
        target_size: Target number of samples
        random_state: Random seed for reproducibility
    
    Returns:
        Undersampled dataframe for the class
    """
    class_df = df[df['dx'] == label]
    return class_df.sample(n=target_size, random_state=random_state)


def combine_datasets(original_df, augmented_df, undersampled_df):
    """
    Combine original (excluding undersampled), undersampled, and augmented data.
    
    Args:
        original_df: Original training dataframe
        undersampled_df: Undersampled class dataframe
        augmented_df: Augmented dataframe
    
    Returns:
        Combined dataframe
    """
    
    non_undersampled = original_df[original_df['dx'] != undersampled_df['dx'].iloc[0]]
    
    return pd.concat(
        [non_undersampled, undersampled_df, augmented_df],
        ignore_index=True
    )


def run_augmentation_pipeline(train_df, resize_strategy, undersample, undersample_size, output_path, augmentation_map=augmentation_multiplier_map, undersample_label='nv'):
    """
    Complete augmentation and balancing pipeline.
    
    Args:
        train_df: Original training dataframe
        augmentation_map: {label: multiplier} for augmentation
        undersample: Boolean to undersample
        undersample_label: Class to undersample (e.g., 'nv')
        undersample_size: Target size for undersampling
        output_path: Path to save augmented_metadata.csv
    
    Returns:
        Combined and balanced dataframe
    """
    print("=" * 70)
    print("AUGMENTATION PIPELINE")
    print("=" * 70)
    
    # Step 1: Generate augmented images
    print("\n[1/3] Generating augmented images...")
    augmented_df = generate_augmented_dataset(train_df, augmentation_map, resize_strategy)
    print(f"Generated {len(augmented_df)} augmented images")
    
    
    # Step 2: Undersample specified class
    if undersample:
        print(f"\n[2/3] Undersampling '{undersample_label}' to {undersample_size} samples...")
        undersampled_df = random_undersample_class(train_df, undersample_label, undersample_size)
        print(f"Undersampled to {len(undersampled_df)} samples")
        
        print("\n[3/3] Combining datasets...")
        aug_train_df = combine_datasets(train_df, augmented_df, undersampled_df)
        print(f"Final dataset size: {len(aug_train_df)} samples")

    else:
        print("\n[3/3] Combining datasets...")
        aug_train_df =  pd.concat([train_df, augmented_df], ignore_index=True)
        print(f"Final dataset size: {len(aug_train_df)} samples")

        
    # Save and report
    if output_path:
        aug_train_df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("FINAL CLASS DISTRIBUTION")
    print("=" * 70)
    print(aug_train_df["dx"].value_counts().sort_index())
    print("=" * 70)

    aug_train_df["dx_encoded"] = aug_train_df["dx"].map(label2idx).astype(int)
    
    return aug_train_df