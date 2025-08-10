import torch
from torch import nn
from ..loss import (
    SSIMLoss, VGG, SSIM_FOCAL, FOCAL_L1, SSIM_L1,
    FOCAL_Char, SSIM_FFL, mse_FFL, Laplace_Loss,
    SSIM_Laplace, Laplace, L2_Total
)

# Registry mapping loss names to their corresponding classes
LOSS_REGISTRY = {
    "SSIMLoss": SSIMLoss,
    "VGG": VGG,
    "SSIM_FOCAL": SSIM_FOCAL,
    "FOCAL_L1": FOCAL_L1,
    "SSIM_L1": SSIM_L1,
    "FOCAL_Char": FOCAL_Char,
    "SSIM_FFL": SSIM_FFL,
    "mse_FFL": mse_FFL,
    "Laplace_Loss": Laplace_Loss,
    "SSIM_Laplace": SSIM_Laplace,
    "Laplace": Laplace,
    "L2_Total": L2_Total
}

def build_loss(config_loss, **extra_kwargs):
    """
    Builds a loss function from the config.

    Args:
        config_loss (dict): Dictionary with 'name' and 'params' keys from the config.
        extra_kwargs: Any extra arguments to pass to the loss class (like noisy_image for L2_Total).

    Returns:
        nn.Module: Instantiated loss function.
    """
    name = config_loss.get("name")
    params = config_loss.get("params", {})

    if name not in LOSS_REGISTRY:
        raise ValueError(f"Loss '{name}' not found in LOSS_REGISTRY. Available: {list(LOSS_REGISTRY.keys())}")

    # Merge config params with extra kwargs (so you can dynamically add args at runtime)
    return LOSS_REGISTRY[name](**{**params, **extra_kwargs})
