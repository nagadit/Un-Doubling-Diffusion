import torch
from diffusers import UNet2DConditionModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def get_torch_dtype(dtype_name):
    """Get PyTorch data type by name"""
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return dtypes.get(dtype_name)


def load_custom_unet(base_model, repo, ckpt, device="cuda", dtype=torch.float16):
    """Helper function to load custom UNet model

    Args:
        base_model: Base model path or identifier
        repo: Repository name
        ckpt: Checkpoint file name
        device: Device to load model to (default: "cuda")
        dtype: Data type for model (default: torch.float16)

    Returns:
        UNet2DConditionModel: Loaded UNet model
    """
    unet = UNet2DConditionModel.from_config(base_model, subfolder="unet").to(device, dtype)
    ckpt_path = hf_hub_download(repo, ckpt)
    unet.load_state_dict(load_file(ckpt_path, device=device))
    return unet
