import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_func(outputs, name, path, figsize=(6, 6)):
    # save results
    with open(os.path.join(path, name + "_res.pkl"), "wb") as fout:
        pickle.dump(outputs, fout)
        
    # generate plot
    matplotlib.rcParams["lines.linewidth"] = 2
    matplotlib.rcParams["font.size"] = 16
    matplotlib.rcParams["font.weight"] = "bold"
    matplotlib.rcParams["font.serif"] = "Times New Roman"

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(outputs["targets"], outputs["preds"], facecolor="None", edgecolors="steelblue", s=100)
    ax.plot([0, 1], [0, 1], linestyle="--", color="red")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis("equal")
    ax.grid(linestyle="--", alpha=0.5)
    
    ax.set_xlabel("Observation", weight="bold")
    ax.set_ylabel("Prediction", weight="bold")
    
    image_file = os.path.join(path, name+".png") 
    plt.savefig(image_file, bbox_inches="tight")