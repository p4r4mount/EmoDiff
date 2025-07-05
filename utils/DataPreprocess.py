import os
import numpy as np
import torch
from event_utils import VoxelGrid
from torchvision.transforms import Resize
from tqdm import tqdm
import pandas as pd
import argparse

def save_voxel(subset_dir):
    fps = 60
    num_bins = 3
    width = 1280
    height = 720
    device = 'cpu'

    subset_name = os.path.basename(subset_dir)
    csv_path = os.path.join(subset_dir, f"{subset_name}.csv")
    voxel_path = os.path.join(subset_dir, f"{subset_name}_voxel_crop.npy")

    df = pd.read_csv(csv_path)
    event_data = df.values.astype(np.int32)

    x = event_data[:, 0]
    y = event_data[:, 1]
    p = event_data[:, 2]
    t = event_data[:, 3]
    t -= t[0]

    delta_t = 1e6 / fps
    num_frames = int(t[-1] / delta_t)

    resize_op = Resize([180, 180], antialias=True)
    voxel_list = []
    

    for idx in tqdm(range(num_frames)):
        start_t = idx * delta_t
        end_t = (idx + 1) * delta_t

        start_idx = np.argmin(np.abs(t - start_t))
        end_idx = np.argmin(np.abs(t - end_t))

        x_slice = x[start_idx:end_idx].astype(np.float32)
        y_slice = y[start_idx:end_idx].astype(np.float32)
        p_slice = p[start_idx:end_idx].astype(np.float32)
        t_slice = t[start_idx:end_idx].astype(np.float32)

        events = np.stack([x_slice, y_slice, t_slice, p_slice], axis=1)
        events = torch.from_numpy(events).to(device)

        voxel_grid = VoxelGrid((num_bins, height, width), normalize=True, device=device)
        voxel = voxel_grid.convert({
            'x': events[:, 0],
            'y': events[:, 1],
            't': events[:, 2],
            'p': events[:, 3],
        })

        voxel_cropped = voxel[:, :, 280:1000]
        voxel_resized = resize_op(voxel_cropped).numpy()
        voxel_list.append(voxel_resized)

    voxel_array = np.array(voxel_list, dtype=np.float16)
    np.save(voxel_path, voxel_array)


def process_dataset(root_dir):
    for split in ['train', 'test', 'val']:
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            continue

        for subfolder in os.listdir(split_dir):
            subfolder_path = os.path.join(split_dir, subfolder)
            if os.path.isdir(subfolder_path):
                try:
                    save_voxel(subfolder_path)
                except Exception as e:
                    print(f"Error processing {subfolder_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run save_voxel on dataset splits.")
    parser.add_argument("dataset_dir", type=str, help="Path to dataset root directory")
    args = parser.parse_args()

    process_dataset(args.dataset_dir)