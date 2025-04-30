import os
import pickle
import shutil
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from utilities.brightness_augmentation import BrightnessAugmentation
from utilities.predictor import LinearModel, predictor
from utilities.Bayesion_conv_layer import BayesianConvLayer
from utilities.Bayesian_linear_layer import BayesianLinearLayer

from utilities.sat_latent_dataset import SatLatentDataset
from utilities.daily_sat_latent_dataset import DailySatLatentDataset
from utilities.plot_func import plot_func
from utilities.daily_plot_func import daily_plot_func


# create Bayesian CNN model
class BayesianCNN(pl.LightningModule):
    """Bayesian CNN model for image regression with uncertainty quantification"""
    def __init__(
        self, 
        in_channels=4,  # input channels number 
        n_monte_carlo=100,  # sampling number
        learning_rate=1e-3,  # learning rate
        alpha=1,  # latent mse loss weight
        beta=1,  #  canopy cover mse loss weight
        kl_weight=0.1,  # KL divergence loss weight
        predictor=predictor,  # canopy cover prediction function
    ):
        super(BayesianCNN, self).__init__()
        self.in_channels = in_channels
        self.n_monte_carlo = n_monte_carlo
        self.learning_rate = learning_rate
        self.alpha = 1
        self.beta = 1
        self.kl_weight = kl_weight
        self.brightness_augmentation = BrightnessAugmentation()

        self.predictor = predictor().to(self.device)
        for param in self.predictor.parameters():
            param.requires_grad = False
        
        self.save_hyperparameters()

        # create Bayesian CNN model
        self.conv1 = BayesianConvLayer(in_channels=self.in_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = BayesianConvLayer(in_channels=16, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = BayesianConvLayer(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat_size = 128 * 5 * 5
        self.fc1 = BayesianLinearLayer(self.flat_size, 1024)
        self.fc2 = BayesianLinearLayer(1024, 512)
        self.fc3 = BayesianLinearLayer(512, 384)


    def forward(self, x):
        x = self.pool(F.gelu(self.conv1(x)))
        x = self.pool(F.gelu(self.conv2(x)))
        x = F.gelu(self.conv3(x))

        x = x.view(-1, self.flat_size)  # flatten

        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 1, 384)
        return x

    def kl_divergence(self):
        """Calculate the total KL divergence of the model"""
        kl_divs = [
            self.conv1.kl_divergence(),
            self.conv2.kl_divergence(),
            self.conv3.kl_divergence(),
            self.fc1.kl_divergence(),
            self.fc2.kl_divergence(),
        ]
        return sum(kl_divs)

    def training_step(self, batch, batch_idx):
        inputs, latents, targets = batch

        # data augmentation
        inputs = self.brightness_augmentation(inputs)

        # prediction latent output and based on latent output generate canopy cover
        latent_outputs = self(inputs)
        outputs = self.predictor(latent_outputs)

        # calculate losses
        latent_loss = F.mse_loss(latent_outputs, latents)
        target_loss = F.mse_loss(outputs.view_as(targets), targets)

        # Scale KL divergence by the number of batches
        kl_div = self.kl_divergence()
        n_batches = len(self.trainer.train_dataloader)
        kl_weight = self.kl_weight / n_batches if n_batches > 0 else self.kl_weight

        train_loss = self.alpha * latent_loss + self.beta * target_loss + kl_weight * kl_div + (self.alpha + self.beta + kl_weight)

        # Log metrics
        self.log("train_loss", train_loss, prog_bar=True)
        self.log("latent loss", latent_loss, prog_bar=True)
        self.log("target loss", target_loss, prog_bar=True)
        self.log("kl div", kl_div, prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        inputs, latents, targets = batch

        # Monte Carlo sampling for predictions
        mc_latent_outputs = []
        for _ in range(self.n_monte_carlo):
            latent_outputs = self(inputs)
            mc_latent_outputs.append(latent_outputs)
        mc_latent_outputs = torch.stack(mc_latent_outputs, dim=0)

        # calculate the mean and variance
        mean_latent_outputs = mc_latent_outputs.mean(dim=0)
        var_latent_outputs = mc_latent_outputs.var(dim=0)
        
        # calculate loss
        outputs = self.predictor(mean_latent_outputs)
        latent_loss = F.mse_loss(mean_latent_outputs, latents)
        target_loss = F.mse_loss(outputs.view_as(targets), targets)
        val_loss = latent_loss + target_loss

        # Log metrics
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_uncertainty", var_latent_outputs.mean(), prog_bar=True)

        return {"val_loss": val_loss}
    
    def test_step(self, batch, batch_idx):
        inputs, latents, targets = batch

        mc_latent_outputs = []
        for _ in range(self.n_monte_carlo):
            latent_outputs = self(inputs)
            mc_latent_outputs.append(latent_outputs)

        mc_latent_outputs = torch.stack(mc_latent_outputs, dim=0)
        mean_latent_outputs = mc_latent_outputs.mean(dim=0)
        var_latent_outputs = mc_latent_outputs.var(dim=0)

        # calculate loss
        outputs = self.predictor(mean_latent_outputs)
        latent_loss = F.mse_loss(mean_latent_outputs, latents)
        target_loss = F.mse_loss(outputs.view_as(targets), targets)
        test_loss = latent_loss + target_loss

        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_uncertainty", var_latent_outputs.mean(), prog_bar=True)
        
        return {"test_loss": test_loss}


    def predict_with_uncertainty(self, x):
        """
        Make predictions with uncertainty quantification

        Args:
            x: Input data of shape (batch_size, 4, 20, 20)

        Returns:
            mean_pred: Mean prediction of shape (batch_size, 1, 384)
            var_pred: Variance of prediction of shape (batch_size, 1, 384)
        """
        # Switch to evaluation mode
        self.eval()
        x = x.to(self.device)

        with torch.no_grad():
            outputs = []
            for _ in range(self.n_monte_carlo):
                latent_outputs = self(x)
                output = self.predictor(latent_outputs)
                outputs.append(output)
            outputs = torch.stack(outputs, dim=0)

            final_output = outputs.mean(dim=0)
            final_uncertainty = outputs.var(dim=0)

        return final_output, final_uncertainty

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        }


if __name__ == "__main__":
    # for reproducibility
    pl.seed_everything(73)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator().manual_seed(42)

    ######################################################################################
    EXPERIMENT_NAME = "nir"
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    MAX_EPOCHS = 200
    N_MONTE_CARLO = 200
    ALPHA = 1
    BETA = 1
    BANDS = [3]
    IN_CHANNELS = len(BANDS)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATHES = {
        "LOGS": "logs",
        "RESULT": "result",
        "SAVED_MODEL": "saved_model"
    }

    DATASETS = {
        "DRISCOLL_2023": {
            "dataset_file": "dataset/latent_dataset.h5",
            "minmax_file": "dataset/driscoll_sat_minmax_2023.pkl",
            "year": "2023",
            "location": "driscoll",
            "bands": BANDS,
            },
        "SINTON_2023": {
            "dataset_file": "dataset/latent_dataset.h5",
            "minmax_file": "dataset/sinton_sat_minmax_2023.pkl",
            "year": "2023",
            "location": "sinton",
            "bands": BANDS,
            },
    }
    ######################################################################################

    # create experiment environment
    for path in PATHES:
        expirement_path = os.path.join(PATHES[path], EXPERIMENT_NAME)
        if os.path.exists(expirement_path):
            shutil.rmtree(expirement_path)
        os.makedirs(expirement_path)
        PATHES[path] = expirement_path

    LOGGER = TensorBoardLogger(PATHES["LOGS"])

    # creater trainer
    trainer = pl.Trainer(
        logger=LOGGER,
        accelerator="gpu" if str(DEVICE).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=MAX_EPOCHS,
        callbacks=[
            ModelCheckpoint(monitor="val_loss", mode="min"),
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            LearningRateMonitor("epoch"),
            RichProgressBar(),
        ]
    )

    # prepare for training
    model = BayesianCNN(
        in_channels=IN_CHANNELS,
        n_monte_carlo=N_MONTE_CARLO,
        learning_rate=LEARNING_RATE,
        alpha=ALPHA,
        beta=BETA,
        )
    
    # load data (only use sinton 2023 for training)
    sinton_2023 = SatLatentDataset(**DATASETS["SINTON_2023"])

    train_set, valid_set, test_set = random_split(sinton_2023, [0.7, 0.15, 0.15], generator=generator)
    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    torch.set_float32_matmul_precision("medium")
    print("[INFO] start training ...")
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader)

    # save the best model
    model = BayesianCNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    model_file = os.path.join(PATHES["SAVED_MODEL"],  EXPERIMENT_NAME + "_model.pt")
    torch.save(model, model_file)
    print("[INFO] Done !!")
    
    ######################################################################################
    # inference test dataset
    outputs = { "preds": [], "uncertainty": [], "targets": [], }
    for inputs, latents, targets in tqdm(test_dataloader):
        pred, uncertainty = model.predict_with_uncertainty(inputs)
        outputs["preds"].append(pred)
        outputs["uncertainty"].append(uncertainty)
        outputs["targets"].append(targets)

    for key in outputs:
        outputs[key] = torch.cat(outputs[key], dim=0)
        outputs[key] = outputs[key].view(-1, 1).detach().cpu().numpy()

    plot_func(outputs, "test_set", PATHES["RESULT"])
    

    ######################################################################################
    # driscoll 2023
    driscoll_2023 = SatLatentDataset(**DATASETS["DRISCOLL_2023"])
    data_loader = DataLoader(driscoll_2023, batch_size=128, shuffle=False, num_workers=4)

    outputs = { "preds": [], "uncertainty": [],  "targets": [] }
    for inputs, latents, targets in tqdm(data_loader):
        pred, uncertainty = model.predict_with_uncertainty(inputs)
        outputs["preds"].append(pred)
        outputs["uncertainty"].append(uncertainty)
        outputs["targets"].append(targets)

    for key in outputs:
        outputs[key] = torch.cat(outputs[key], dim=0)
        outputs[key] = outputs[key].view(-1, 1).detach().cpu().numpy()

    plot_func(outputs, "driscoll_2023", PATHES["RESULT"])

    ######################################################################################
    # generate daily result
    ## driscoll 2023
    daily_plot_func(
        model=model,
        dataset_file=DATASETS["DRISCOLL_2023"]["dataset_file"],
        minmax_file=DATASETS["DRISCOLL_2023"]["minmax_file"],
        location="driscoll",
        bands=BANDS,
        path=PATHES["RESULT"]
    )

    ## sinton 2023
    daily_plot_func(
        model=model,
        dataset_file=DATASETS["SINTON_2023"]["dataset_file"],
        minmax_file=DATASETS["SINTON_2023"]["minmax_file"],
        location="sinton",
        bands=BANDS,
        path=PATHES["RESULT"]
    )

    print("[INFO] All Done !!")