"""
Data augmenetation:
    randomly change the brightness of RGB bands
    According to Ashutosh, NIR should not be impacted by the brightness
"""
import numpy as np

import torch
import torch.nn as nn


class BrightnessAugmentation(nn.Module):
    def forward(self, image):
        brightness = np.random.uniform(0.8, 1.2)
        image[:, :3, :, :] =  image[:, :3, :, :] * brightness
        return image


if __name__ == "__main__":
    from sat_latent_dataset import SatLatentDataset
    from torch.utils.data import DataLoader
    from torchvision import transforms

    dataset_file = "dataset/latent_dataset.h5"
    brightness_augmentation = BrightnessAugmentation()    

    # driscoll 2023
    minmax_file = "dataset/driscoll_sat_minmax_2023.pkl"
    driscoll_2023 = SatLatentDataset(dataset_file, minmax_file)
    driscoll_loader_2023 = DataLoader(driscoll_2023, batch_size=1, shuffle=True, num_workers=4)
    images, latents, targets = next(iter(driscoll_loader_2023))

    print("*" * 100)
    print("drsicoll 2023")
    print(f"images shape: {images.shape}, type: {type(images)}, dtype: {images.dtype}")
    print(f"image minimum: {images.min()}, maximum: {images.max()}")
    print(f"latents shape: {latents.shape}, type: {type(latents)}, dtype: {latents.dtype}")
    print(f"targets shape: {targets.shape}, type: {type(targets)}, dtype: {targets.dtype}")

    print(f"\ndata size: {len(driscoll_2023)}")
    print("*" * 100)
    
    print(images[0, -1, :, :])
    images = brightness_augmentation(images)
    print(images[0, -1, :, :])


