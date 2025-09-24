import torch

import os
import inspect
import hydra
from hydra.utils import instantiate

from tqdm import tqdm

import pandas as pd


@hydra.main(config_path="configs/", config_name="seq", version_base=None)
def main(cfg) -> None:
    """
    Generate cfg.num_images images one by one (prevent CUDA OOM).
    The duration of generations (cfg.num_images * len(homonyms) * len(models)) is reduced by parallel script execution
    (for each model).
    """

    seeds = torch.arange(end=cfg.num_images)

    homonyms = pd.read_csv("../annotation.tsv", sep="\t")["homonym"].unique().tolist()

    with torch.no_grad():
        # generations will only be performed for the assigned model
        modeln = cfg.modelname4generation
        pipe = instantiate(cfg.models[modeln])

        # take num_inference_steps the same as in the signature
        num_inference_steps = dict(inspect.signature(pipe).parameters)['num_inference_steps'].default
        # gs is also taken from the signature
        guidance_scale = dict(inspect.signature(pipe).parameters)['guidance_scale'].default

        pipe = pipe.to('cuda')
        pipe.set_progress_bar_config(leave=False)  # remove progress bar over num_inference_steps after completion
        pipe.enable_model_cpu_offload()  # pipeline is offloaded to cpu when not in use

        pbar = tqdm(homonyms)
        for prompt in pbar:
            pbar.set_description(f"{modeln} / {prompt}")

            save_dir = os.path.join(cfg.save_path, modeln, prompt)
            os.makedirs(save_dir, exist_ok=True)

            for i in tqdm(range(cfg.num_images)):
                seed = seeds[i].item()
                images = pipe(
                    prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=1,
                    height=1024, width=1024, max_sequence_length=256,  # fix among models
                    generator=torch.Generator("cpu").manual_seed(seed)
                ).images

                img = images[0]
                img.save(os.path.join(save_dir, f"{i}.png"))


if __name__ == "__main__":
    main()
