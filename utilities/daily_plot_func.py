import os
import h5py
import pickle
import shutil

from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader
from utilities.daily_sat_latent_dataset import DailySatLatentDataset
from utilities.plot_func import plot_func


def daily_plot_func(model, dataset_file, minmax_file, location, bands, path):
    # create result path
    res_folder = "daily" + "_" + location
    res_path = os.path.join(path, res_folder)
    if os.path.exists(res_path):
        shutil.rmtree(res_path)
    os.makedirs(res_path)
    
    dataset = h5py.File(dataset_file)
    sat_dates = sorted(dataset["2023"][location]["sat"].keys())
    latent_dates = sorted(dataset["2023"][location]["cc"].keys())
    dataset.close()
    
    if location == "sinton":
        idx = sat_dates.index("20230415")
        sat_dates.pop(idx)
        latent_dates.pop(idx)

        idx = sat_dates.index("20230510")
        sat_dates.pop(idx)
        latent_dates.pop(idx)
     
    for sat_date, latent_date in tqdm(zip(sat_dates, latent_dates)):
        dataset = DailySatLatentDataset(dataset_file, minmax_file, sat_date, latent_date, location=location, bands=bands)
        data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
        
        # inference
        outputs = {"fids": [], "preds": [], "uncertainty": [], "targets": []}

        for inputs, targets, fids in tqdm(data_loader):
            pred, uncertainty = model.predict_with_uncertainty(inputs)
            outputs["preds"].append(pred)
            outputs["uncertainty"].append(uncertainty)
            outputs["targets"].append(targets)
            outputs["fids"].append(targets)
            
        for key in outputs:
            outputs[key] = torch.cat(outputs[key], dim=0)
            outputs[key] = outputs[key].view(-1, 1).detach().cpu().numpy()           

        # plot and save results
        plot_func(outputs, sat_date, res_path)
        










    


    # plot configuration
    # matplotlib.rcParams["lines.linewidth"] = 2
    # matplotlib.rcParams["font.size"] = 16
    # matplotlib.rcParams["font.weight"] = "bold"
    # matplotlib.rcParams["font.serif"] = "Times New Roman"     
    
    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.scatter(observation, prediction, facecolor="None", edgecolors="steelblue", s=100)
    # ax.plot([0, 1], [0, 1], linestyle="--", color="red")
    # ax.set_xlim([0, 1])
    # ax.set_ylim([0, 1])
    # ax.axis("equal")
    # ax.grid(linestyle="--", alpha=0.5)
    
    # ax.set_xlabel("Observation", weight="bold")
    # ax.set_ylabel("Prediction", weight="bold")
    
    # image_file = os.path.join(path, name+".png") 
    # plt.savefig(image_file, bbox_inches="tight")
    