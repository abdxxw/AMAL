import optuna 
from optuna.integration import PyTorchLightningPruningCallback

import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,random_split
from pathlib import Path
from datamaestro import prepare_dataset
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from tp7_lightning import LitMnistData, Lit2Layer

from torchvision import transforms

BATCH_SIZE = 300
TRAIN_RATIO = 0.05
LOG_PATH = "/tmp/runs/lightning_logs"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def objective(trial: optuna.trial.Trial) -> float:


    EPOCHS = 1000
    PERCENT_VALID_EXAMPLES = 0.2

    batch_norm = trial.suggest_categorical("batch_norm", [True, False])
    layer_norm = trial.suggest_categorical("layer_norm", [True, False])
    dropout = trial.suggest_float("dropout", 0,1)
    l1_reg = trial.suggest_categorical("l1_reg", [0, 0.0001, 0.001, 0.005, 0.05, 0.01])
    l2_reg = trial.suggest_categorical("l2_reg", [0, 0.0001, 0.001, 0.005, 0.05, 0.01])
    
    data = LitMnistData()


    model = Lit2Layer(data.dim_in,data.dim_out,learning_rate=1e-3,dropout=dropout,l1=l1_reg,l2=l2_reg,batchnorm=batch_norm,layernorm=layer_norm)
    

    trainer = pl.Trainer(
        limit_val_batches=PERCENT_VALID_EXAMPLES,
        logger=True,
        checkpoint_callback=False,
        max_epochs=EPOCHS,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_accuracy")],
    )

    hyperparameters = dict(dropout=dropout,l1=l1_reg,l2=l2_reg,batchnorm=batch_norm,layernorm=layer_norm)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=data)

    return trainer.callback_metrics["val_accuracy"].item()




study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20, timeout=60000)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    
res = study.trials_dataframe()

res = res[["params_batch_norm","params_layer_norm","params_dropout","params_l1_reg","params_l2_reg","value"]].sort_values("value",ascending=False)

print(res)