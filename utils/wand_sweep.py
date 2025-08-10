import wandb
import itertools
#This contains the various configs for W&B runs

ssim_laplace_sweep_config = {
    "method": "grid",  # Use grid search instead of random
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "lr": {
            "values": [
                    0.001
            ]
        },
        "w": {
            # "values": list(np.arange(0.4, 0.75,0.03))
            "values": [0.51]
        }
    },
}

ssim_fdl_sweep_config = {
    "method": "grid",  # Use grid search instead of random
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "lr": {
            "values": [
                    0.001
            ]
        },
        "w": {
            # "values": list(np.arange(0.4, 0.75,0.03))
            "values": [0.50]
        }
    },
}
#REMEMBER THAT SSIM_LAPLACE ALREADY HAS MODELS SAVED TILL 16TH EPOCH

char_fdl_sweep_config = {
    "method": "grid",  # Use grid search instead of random
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "lr": {
            "values": [
                    0.0001
            ]
        },
        "w": {
            # "values": list(np.arange(0.4, 0.75,0.03))
            "values": [0.51]
        }
    },
}
ssim_ffl_sweep_config = {
    "method": "grid",  # Use grid search instead of random
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "lr": {
            "values": [
                    0.001
            ]
        },
        "w": {
            # "values": list(np.arange(0.4, 0.75,0.03))
            "values": [0.6]
        }
    },
}


config = ssim_laplace_sweep_config

total_combinations = len(list(itertools.product(*config['parameters'].values())))
print("Total Combinations of the chosen parameters:",total_combinations)
sweep_id = wandb.sweep(config, project="Train_SSIM_FDL_EffNet")
print(f"Created Sweep with ID: {sweep_id}")
