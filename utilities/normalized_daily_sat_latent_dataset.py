"""create sat cc dataset"""
import h5py
import pickle
import numpy as np

from torch.utils.data import Dataset


class DailySatLatentDataset(Dataset):
    def __init__(
            self,
            dataset_file,
            minmax_file,
            sat_date,
            latent_date,
            year = "2023",
            location = "driscoll",
            bands = [0, 1, 2, 3],
            transform = None,
    ):
        super(DailySatLatentDataset, self).__init__()
        self.minmax_file = minmax_file
        self.sat_date = sat_date
        self.latent_date = latent_date
        self.bands = bands
        self.transform = transform
        self.dataset = h5py.File(dataset_file)

        # create dataset
        self.fids = sorted(self.dataset[year][location]["sat"][self.sat_date].keys())
        self.sat_images = [self.dataset[year][location]["sat"][self.sat_date][fid] for fid in self.fids]
        self.targets = [self.dataset[year][location]["cc"][self.latent_date][fid] for fid in self.fids]

    def __len__(self):
        return len(self.sat_images)
    
    def __getitem__(self, idx):
        fid = self.fids[idx] 
        sat_image = self.sat_images[idx]
        target = self.targets[idx]
        target = np.array(target, dtype=np.float32)

        sat_image = self._minMax(np.array(sat_image, dtype=np.float32), self.sat_date)
        # normalized by gb_nir
        sat_image = sat_image / (sat_image[1:, :, :].sum(axis=0) + 1e-6)

        temp = []
        for band in self.bands:
            temp.append(sat_image[band])
        sat_image = np.stack(temp)

        if self.transform:
            sat_image = self.transform(sat_image)

        return sat_image, target, fid
    
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
    dataset = h5py.File(dataset_file) 

    # driscoll 2023
    sat_dates = sorted(dataset["2023"]["driscoll"]["sat"].keys())
    latent_dates = sorted(dataset["2023"]["driscoll"]["latent"].keys())
    minmax_file = "dataset/driscoll_sat_minmax_2023.pkl"
    
    driscoll_2023 = DailySatLatentDataset(dataset_file, minmax_file, sat_dates[0], latent_dates[0])
    data_loader = DataLoader(driscoll_2023, batch_size=16, shuffle=True, num_workers=4)
    images, targets, fids = next(iter(data_loader))
    
    print("*" * 100)
    print("drsicoll 2023")
    print(f"images shape: {images.shape}, type: {type(images)}, dtype: {images.dtype}")
    print(f"image minimum: {images.min()}, maximum: {images.max()}")
    print(f"targets shape: {targets.shape}, type: {type(targets)}, dtype: {targets.dtype}")
    print(f"data size: {len(driscoll_2023)}")
    print(f"fids: {fids}")
    print("*" * 100)


    # sinton 2023
    sat_dates = sorted(dataset["2023"]["sinton"]["sat"].keys())
    latent_dates = sorted(dataset["2023"]["sinton"]["latent"].keys())
    minmax_file = "dataset/sinton_sat_minmax_2023.pkl"

    sinton_2023 = DailySatLatentDataset(dataset_file, minmax_file, sat_dates[0], latent_dates[0], location="sinton")
    data_loader = DataLoader(sinton_2023, batch_size=16, shuffle=True, num_workers=4)
    images, targets, fids = next(iter(data_loader))

    print("*" * 100)
    print("drsicoll 2024")
    print(f"images shape: {images.shape}, type: {type(images)}, dtype: {images.dtype}")
    print(f"image minimum: {images.min()}, maximum: {images.max()}")
    print(f"targets shape: {targets.shape}, type: {type(targets)}, dtype: {targets.dtype}")
    print(f"data size: {len(sinton_2023)}")
    print(f"fids: {fids}")
    print("*" * 100)
 