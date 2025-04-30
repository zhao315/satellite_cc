"""latent dataset: cc, sat, latent"""
import h5py
import pickle
import numpy as np

from torch.utils.data import Dataset


class SatLatentDataset(Dataset):
    def __init__(
        self, 
        dataset_file,  # total dataset 
        minmax_file, # minmax by day 
        year = "2023", 
        location = "driscoll",
        bands = [0, 1, 2, 3],
        transform = None
        ):
        super(SatLatentDataset, self).__init__()
        self.minmax_file = minmax_file
        self.bands = bands
        self.transform = transform
        self.dataset = h5py.File(dataset_file)

        # create dataset
        self.sat_images = []
        self.latent_vectors = []
        self.targets = []
        self.dates = []
            
        self.sat_dates = sorted(self.dataset[year][location]["sat"].keys())
        self.latent_dates = sorted(self.dataset[year][location]["latent"].keys())

        if location == "sinton":
            idx = self.sat_dates.index("20230415")
            self.sat_dates.pop(idx)
            self.latent_dates.pop(idx)

            idx = self.sat_dates.index("20230510")
            self.sat_dates.pop(idx)
            self.latent_dates.pop(idx)

        for sat_date, latent_date in zip(self.sat_dates, self.latent_dates):
            fids = sorted(self.dataset[year][location]["sat"][sat_date].keys())
            sat_image = [self.dataset[year][location]["sat"][sat_date][fid] for fid in fids]
            latent_vector = [self.dataset[year][location]["latent"][latent_date][fid] for fid in fids]
            target = [self.dataset[year][location]["cc"][latent_date][fid] for fid in fids]
            dates = [sat_date] * len(fids)
            
            self.sat_images.extend(sat_image)
            self.latent_vectors.extend(latent_vector)
            self.targets.extend(target)
            self.dates.extend(dates)

    def __len__(self):
        return len(self.sat_images)
    
    def __getitem__(self, idx):
        latent_vector = self.latent_vectors[idx]
        target = self.targets[idx]
        sat_image = self.sat_images[idx]
        date = self.dates[idx]

        latent_vector = np.array(latent_vector, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        
        sat_image = self._minMax(np.array(sat_image, dtype=np.float32), date)

        # normalized by gb_nir
        sat_image = sat_image / (sat_image[1:, :, :].sum(axis=0) + 1e-6)

        # create sat image data
        temp = []
        for band in self.bands:
            temp.append(sat_image[band])
        sat_image = np.stack(temp)

        if self.transform:
            sat_image = self.transform(sat_image)

        return sat_image, latent_vector, target
    
    def _minMax(self, data, date):
        with open(self.minmax_file, "rb") as fin:
            minmax_value = pickle.load(fin)

        data[0] = (data[0] - minmax_value[date]["r"]["min"]) / (minmax_value[date]["r"]["max"] - minmax_value[date]["r"]["min"])
        data[1] = (data[1] - minmax_value[date]["g"]["min"]) / (minmax_value[date]["g"]["max"] - minmax_value[date]["g"]["min"])
        data[2] = (data[2] - minmax_value[date]["b"]["min"]) / (minmax_value[date]["b"]["max"] - minmax_value[date]["b"]["min"])
        data[3] = (data[3] - minmax_value[date]["nir"]["min"]) / (minmax_value[date]["nir"]["max"] - minmax_value[date]["nir"]["min"])

        return data


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset_file = "dataset/latent_dataset.h5"
    
    # driscoll 2023
    minmax_file = "dataset/driscoll_sat_minmax_2023.pkl"
    driscoll_2023 = SatLatentDataset(dataset_file, minmax_file)
    driscoll_loader_2023 = DataLoader(driscoll_2023, batch_size=16, shuffle=True, num_workers=4)
    images, latents, targets = next(iter(driscoll_loader_2023))

    print("*" * 100)
    print("drsicoll 2023")
    print(f"images shape: {images.shape}, type: {type(images)}, dtype: {images.dtype}")
    print(f"image minimum: {images.min()}, maximum: {images.max()}")
    print(f"latents shape: {latents.shape}, type: {type(latents)}, dtype: {latents.dtype}")
    print(f"targets shape: {targets.shape}, type: {type(targets)}, dtype: {targets.dtype}")

    print(f"\ndata size: {len(driscoll_2023)}")
    print("*" * 100)

    # sinton 2023
    minmax_file = "dataset/sinton_sat_minmax_2023.pkl"
    sinton_2023 = SatLatentDataset(dataset_file, minmax_file, location="sinton")
    sinton_loader_2023 = DataLoader(sinton_2023, batch_size=16, shuffle=True, num_workers=4)
    images, latents, targets = next(iter(sinton_loader_2023))

    print("*" * 100)
    print("sinton 2023")
    print(f"images shape: {images.shape}, type: {type(images)}, dtype: {images.dtype}")
    print(f"image minimum: {images.min()}, maximum: {images.max()}")
    print(f"latents shape: {latents.shape}, type: {type(latents)}, dtype: {latents.dtype}")
    print(f"targets shape: {targets.shape}, type: {type(targets)}, dtype: {targets.dtype}")

    print(f"\ndata size: {len(sinton_2023)}")
    print("*" * 100)
 
 