# Un-Doubling Diffusion: LLM-guided Disambiguation of Homonym Duplication 

![.](images/1.png)
## Complete Folder and File Structure

```
.
├── README.md
├── annotation.tsv
├── autoeval/
│   ├── README.md
│   ├── autoeval_shooter.py
│   ├── qwen72b_vllm_server.sh
│   └── utils.py
├── generation/
│   ├── README.md
│   ├── generate_onebyone.py
│   ├── utils.py
│   └── configs/
│       ├── seq.yaml
│       └── models/
│           └── all.yaml
└── prompt_expansion/
    ├── README.md
    └── generate_expanded_prompt.py
```

## Module Details

### Auto Evaluation (autoeval)

This module contains tools for automatically evaluating model performance on homonym-related tasks.

#### Key Scripts:
- `autoeval_shooter.py`: Main evaluation script
- `qwen72b_vllm_server.sh <n_gpu>`: Launch script for Qwen 72B model with vLLM

#### Environment Requirements:
- python==3.9.21
- cuda-toolkit==12.9.1
- cudnn==9.10.1.4
- cuda-nvcc==12.9.86
- cuda==12.9
- torch==2.7.1
- torchvision==0.22.1
- vllm==0.10.0
- transformers==4.53.3
- hydra-core==1.3.2
- xformers==0.0.31

### Homonym Generation (generation)

This module contains scripts for generating homonym-related content.

#### Implementation details
`Sequential generation` is used:
In a loop, the specified number of images is generated, with num_images_per_prompt set to one.
This prevents `CUDA OOM`, while the seed is explicitly assigned each time, incrementing by one
(`range(0, number of images)`). This generation is much slower (1 image is generated at a time instead of batch images), but
with it, all seeds for all generated images are transparent.

#### Environment Requirements:
- python=3.11.5
- flash_attn==2.5.8
- torch==2.3.1
- torchvision==0.18.1
- cuda==2.6.1
- cudnn==8.9.2.26
- cuda-nvcc==12.6.68
- xformers==0.0.27

#### Attention types for model modules
<table>
  <tr>
    <th rowspan="2">Model (dtype)</th>
    <th colspan="3">Text modules</th>
    <th rowspan="2">VAE module</th>
    <th rowspan="2">U-Net / Transformer module</th>
  </tr>

  <tr>
    <th>text_encoder_1</th>
    <th>text_encoder_2</th>
    <th>text_encoder_3</th>
  </tr>

  <tr>
    <td>Stable Diffusion 3 Medium (float16)</td>
    <td>SDPA</td>
    <td>SDPA</td>
    <td>Eager</td>
    <td>SDPA</td>
    <td>SDPA</td>
  </tr>
  <tr>
    <td>Stable Diffusion 3.5 Large (bfloat16)</td>
    <td>SDPA</td>
    <td>SDPA</td>
    <td>Eager</td>
    <td>SDPA</td>
    <td>SDPA</td>
  </tr>
  <tr>
    <td>Stable Diffusion 3.5 Medium (bfloat16)</td>
    <td>SDPA</td>
    <td>SDPA</td>
    <td>Eager</td>
    <td>SDPA</td>
    <td>SDPA</td>
  </tr>
  <tr>
    <td>Stable Diffusion XL (float16)</td>
    <td>SDPA</td>
    <td>SDPA</td>
    <td>None</td>
    <td>SDPA</td>
    <td>SDPA</td>
  </tr>

  <tr>
    <td>FLUX.1-schnell (bfloat16)</td>
    <td>SDPA</td>
    <td>Eager</td>
    <td>None</td>
    <td>SDPA</td>
    <td>SDPA</td>
  </tr>
  <tr>
    <td>FLUX.1-dev (bfloat16)</td>
    <td>SDPA</td>
    <td>Eager</td>
    <td>None</td>
    <td>SDPA</td>
    <td>SDPA</td>
  </tr>

  <tr>
    <td>Kandinsky 3 (float16)</td>
    <td>Eager</td>
    <td>None</td>
    <td>None</td>
    <td>(MoVQ) SDPA</td>
    <td>SDPA</td>
  </tr>

  <tr>
    <td>Pixart Alpha (float16)</td>
    <td>Eager</td>
    <td>None</td>
    <td>None</td>
    <td>SDPA</td>
    <td>SDPA</td>
  </tr>
  <tr>
    <td>Pixart Sigma (float16)</td>
    <td>Eager</td>
    <td>None</td>
    <td>None</td>
    <td>SDPA</td>
    <td>SDPA</td>
  </tr>

  <tr>
    <td>Playground v2.5 (float16)</td>
    <td>SDPA</td>
    <td>SDPA</td>
    <td>None</td>
    <td>SDPA</td>
    <td>SDPA</td>
  </tr>
  <tr>
    <td>CogView 4 (bfloat16)</td>
    <td>SDPA</td>
    <td>None</td>
    <td>None</td>
    <td>w/o attn</td>
    <td>SDPA</td>
  </tr>
</table>

### Prompt Expansion

#### Prompt Expansion (Beautification)
```bash
python generate_expanded_prompt.py
```

#### Environment Requirements:
- python==3.9.21
- cuda-toolkit==12.9.1
- cudnn==9.10.1.4
- cuda-nvcc==12.9.86
- cuda==12.9
- torch==2.7.1
- torchvision==0.22.1
- vllm==0.10.0
- transformers==4.53.3
- hydra-core==1.3.2
- xformers==0.0.31
