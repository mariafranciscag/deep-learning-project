import pandas as pd
import numpy as np
import json
import time
import tensorflow as tf
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import os
import uuid
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

with open("label2idx.json", "r") as f:
    label2idx = json.load(f)

torch.manual_seed(1)

# new images
base_path = "./data" 
aug_dir = os.path.join(base_path, "HAM10000_augmented")
if not os.path.exists(aug_dir):
    os.makedirs(aug_dir)

means = [0.485, 0.456, 0.406]
stds  = [0.229, 0.224, 0.225]


# Augmentation Strategies 

## STRATEGY 1 — Light (geometric)
### safe flips and rotation, nothing that touches color or structure
strategy_1 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize(means, stds)
])


## STRATEGY 2 — Moderate (geometry + mild color)
### adds subtle brightness/contrast to simulate lighting variation
strategy_2 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(means, stds),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.05)),  # simulates hair/occlusion
])


## STRATEGY 3 — Aggressive (everything + elastic + perspective)
### most aggressive — use only for extreme minority classes (vasc, df)
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
    transforms.Normalize(means, stds),
    transforms.RandomErasing(p=0.35, scale=(0.02, 0.08)),
])

### base strategy, for the majority class
strat_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(means, stds),
])


## SAVE STRATEGIES

save_strategy_1 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
])

save_strategy_2 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

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


# Augmentation Maps

def get_square_root_maps(df, target_size=4500):
    counts = df['dx'].value_counts()
    
    mult_map = {}
    strat_map = {}
    save_strat_map = {}
    
    for label, count in counts.items():
        if label == 'nv':
            mult_map[label] = 1
            strat_map[label] = strat_base
            continue
            
        #Square Root Math
        raw_mult = np.sqrt(target_size / count)
        final_mult = int(np.round(raw_mult))
        
        #we always have at least 1
        mult_map[label] = max(1, final_mult)
        
        #strategy assignment
        if mult_map[label] <= 1:
            strat_map[label] = strat_base
            save_strat_map[label] = save_strat_base
        elif mult_map[label] <= 3:
            strat_map[label] = strategy_1 #mild stretch
            save_strat_map[label] = save_strategy_1
        elif mult_map[label] <= 5:
            strat_map[label] = strategy_2 #moderate stretch
            save_strat_map[label] = save_strategy_2
        else:
            strat_map[label] = strategy_3 #heavy stretch
            save_strat_map[label] = save_strategy_3
            
    return mult_map, strat_map, save_strat_map



# Augmentation Pipeline

def augment_single_image(original_path, transform, new_id):
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
    
    if transform:
        img = transform(img)
    
    new_filename = f"{new_id}.jpg"
    save_path = os.path.join(aug_dir, new_filename)
    img.save(save_path, "JPEG", quality=95, subsampling=0)
    
    return save_path


def build_metadata_row(new_id, save_path, label, lesion_id, original_row):
    """
    Build metadata row for augmented image, preserving all clinical data.
    """
    # 1. Create a full copy of the original clinical record (Age, Sex, Loc)
    new_row_dict = original_row.copy()
    
    # 2. Update ONLY the image-specific identifiers for the new augmented version
    new_row_dict.update({
        'image_id': new_id,      
        'image_path': save_path, # Path to the new JPG
        'dx': label,
        'dataset': 'augmented',
        'lesion_id': lesion_id
    })
    
    return new_row_dict


def offline_augmentation(df, label, multiplier, save_strategy_map):
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
    
    #class data
    class_df = df[df['dx'] == label]
    paths = class_df['image_path'].tolist()
    metadata_rows = class_df.to_dict('records')
    
    #calculate augmentations needed
    n_to_generate = int(len(paths) * (multiplier - 1))
    transform = save_strategy_map.get(label)
    
    #pre-generate all UUIDs (2 per augmented image: image_id + lesion_id)
    uuids = [uuid.uuid4().hex[:8] for _ in range(n_to_generate * 2)]
    
    for i in range(n_to_generate):
        idx = i % len(paths)
        original_path = paths[idx]
        original_row = metadata_rows[idx]
        
        #augment image
        new_id = f"AUG_{label}_{uuids[i*2]}"
        save_path = augment_single_image(original_path, transform, new_id)
        
        #build metadata
        lesion_id = f"AUG_LESION_{label}_{uuids[i*2+1]}"
        new_row = build_metadata_row(new_id, save_path, label, lesion_id, original_row)
        new_rows.append(new_row)
    
    return new_rows


def generate_augmented_dataset(df, aug_map, save_strategy_map):
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
        
        #augmentation tasks
        for label, multiplier in aug_map.items():
            if multiplier > 1:
                print(f"Submitting task for augmenting {label} (multiplier: {multiplier})")
                futures.append(
                    executor.submit(offline_augmentation, df, label, multiplier, save_strategy_map)
                )
       
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


def run_augmentation_pipeline(train_df, undersample, undersample_size, output_path, augmentation_map, save_strategy_map, undersample_label='nv'):
    print("=" * 70)
    print("AUGMENTATION PIPELINE (FIXED)")
    print("=" * 70)
    
    # Generate augmented images
    # Every row in augmented_df will ALREADY have age, sex, etc.
    augmented_df = generate_augmented_dataset(train_df, augmentation_map, save_strategy_map)
    print(f"Generated {len(augmented_df)} augmented images")
    
    # Balance the dataset
    if undersample:
        print(f"Undersampling '{undersample_label}'...")
        undersampled_df = random_undersample_class(train_df, undersample_label, undersample_size)
        non_undersampled = train_df[train_df['dx'] != undersample_label]
        aug_train_df = pd.concat([non_undersampled, undersampled_df, augmented_df], ignore_index=True)
    else:
        aug_train_df = pd.concat([train_df, augmented_df], ignore_index=True)

    # Save simple and clean
    if output_path:
        # Final safety check for labels
        aug_train_df["dx_encoded"] = aug_train_df["dx"].map(label2idx).astype(int)
        
        # Save the file - it now contains both ISIC_ and AUG_ rows with full metadata
        aug_train_df.to_csv(output_path, index=False)
        print(f"\nSUCCESS: Saved {len(aug_train_df)} total rows to: {output_path}")

    # VERIFICATION: Show us the augmented rows!
    print("\nSample of Augmented Rows (The bottom of the file):")
    print(aug_train_df.tail(5)[['image_id', 'dx', 'age', 'sex']])
    
    return aug_train_df

def preprocess_imagenet(img):
    """
    Preprocess an image using ImageNet normalization.

    Scales pixel values to [0, 1] and then standardizes each channel
    using the ImageNet mean and standard deviation.

    Args:
        img (np.ndarray): Input image with pixel values in [0, 255].

    Returns:
        np.ndarray: Normalized image ready for a pretrained ImageNet model.
    """
    return (img / 255.0 - means) / stds

#Visualization
#function to denormalize the images, only for visualization purposes
def denormalize(tensor, mean, std):
    #reverse: x = (z * std) + mean
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    img = img * np.array(std) + np.array(mean)
    return np.clip(img, 0, 1)