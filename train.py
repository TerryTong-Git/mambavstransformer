# %% [markdown]
# # Mamba VS Transformer Diffusion

# %%
import random
from collections import namedtuple
from pathlib import Path
from functools import lru_cache

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# %%
from trainloader import Config

# def show(x):
#     if not isinstance(x, torch.Tensor) or x.ndim == 4:
#         x = torch.cat(tuple(x), -1)
#     display(TF.to_pil_image(x))

# %% [markdown]
# # Load Dataset

# %%
# import datasets
# import pyarrow.parquet as pq
# import pyarrow.ipc as ipc
# from io import BytesIO
# import io
# import os
# from tqdm import tqdm
# import numpy as np
# import glob
# def get_dataset(name):
#     # if Path(name).exists():
#     #     print(f"dataset '{name}' already exists; skipping...")
#     #     return
#     # datasets.load_dataset("imagenet-1k")
#     # datasets.save_to_disk("./imagenet-1k")
#     # !git clone https://huggingface.co/datasets/huggan/{name} && (cd {name} && git lfs pull)
    
#     i = 0
#     arrow_files_dir = name + "/train/"
#     output_dir = './data'

#     # Create the output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
#     arrow_files = glob.glob(os.path.join(arrow_files_dir, '*.arrow'))

#     # Iterate over each Arrow file
#     for arrow_file_path in arrow_files:
#     # Read the Arrow file
#         with open(arrow_file_path, 'rb') as f:
#             reader = ipc.RecordBatchStreamReader(f)
#             table = reader.read_all()

#         # Assume the table has columns 'image_data' and 'label'
#         image_data_column = table.column('image')
#         labels_column = table.column('label') if 'label' in table.column_names else None

#         for i in tqdm(range(len(image_data_column))):
#             # Convert the image data to a NumPy array
#             image_data = image_data_column[i]['bytes'].as_py()
#             image_array = np.frombuffer(image_data, dtype=np.uint8)
            
#             # Convert the NumPy array to a PIL Image
#             image = Image.open(io.BytesIO(image_array))
            
#             # Define the output file path
#             output_file_path = os.path.join(output_dir, f'{os.path.basename(arrow_file_path).split(".")[0]}_image_{i}.jpg')
            
#             # Save the image as a JPEG file
#             image.save(output_file_path)
            
#             # Optionally, save the label
#             if labels_column:
#                 label = labels_column[i].as_py()
#                 with open(os.path.join(output_dir, f'{os.path.basename(arrow_file_path).split(".")[0]}_image_{i}.txt'), 'w') as label_file:
#                     label_file.write(str(label))
# get_dataset(Config.dataset)

# %%
Sample = namedtuple("Sample", ("im", "noisy_im", "noise_level"))

def alpha_blend(a, b, alpha):
    return alpha * a + (1 - alpha) * b

@lru_cache(maxsize=10000)
def load_im(path):
    return TF.pil_to_tensor(TF.center_crop(TF.resize(Image.open(path), Config.image_size), Config.image_size).convert("RGB"))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, p):
        self.ims = list(Path(p).rglob("*.jpg")) + list(Path(p).rglob("*.png"))
    def __len__(self):
        return len(self.ims)
    def __getitem__(self, i):
        im = load_im(self.ims[i]) / 255.0
        if random.random() < 0.5:
            im = TF.hflip(im)
        noise = torch.rand_like(im)
        noise_level = torch.rand(1, 1, 1)
        noisy_im = alpha_blend(noise, im, noise_level)
        return Sample(im, noisy_im, noise_level)

d_train = Dataset("data")

# %%


# %%
# def demo_dataset(dataset, n=16):
#     print(f"Dataset has {len(dataset)} samples (not counting augmentation).")
#     print(f"Here are some samples from the dataset:")
#     samples = random.choices(dataset, k=n)
#     print(f"Inputs")
#     # show(s.noisy_im for s in samples)
#     # show(s.noise_level.expand(3, 16, Config.image_size) for s in samples)
#     # print(f"Target Outputs")
#     # show(s.im for s in samples)
# demo_dataset(d_train)

# %%
# def to_device(ims):
#     return Sample(*(x.to(Config.device) for x in ims))

# # make sure the entire dataset loads
# for batch in tqdm(torch.utils.data.DataLoader(d_train, num_workers=64, batch_size=64)): to_device(batch)

# %% [markdown]
# # Model

# %%
from model import MambaDenoiser, TransformerDenoiser
from dataclasses import dataclass, asdict

@dataclass
class DenoiserConfig:
    image_size: int = 64
    noise_embed_dims: int = 32 * 4
    patch_size: int = 2
    embed_dim: int = 32 * 4
    dropout: float = 0
    n_layers: int = 8
    n_channels: int = 3

denoiser_config = DenoiserConfig()

# %%
model = MambaDenoiser(**asdict(denoiser_config)).to(Config.device)

# %%
def weight_average(w_prev, w_new, n):
    alpha = min(0.9, n / 10)
    return alpha_blend(w_prev, w_new, alpha)
    
avg_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=weight_average)

# @torch.no_grad()
# def demo_model(model, n=16):
#     model.eval()
#     n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Model has {n_parameters / 1e6:.1f} million trainable parameters.")
#     x = torch.rand(n, *Config.shape, device=Config.device)
#     noise_level = torch.rand(n, 1, 1, 1, device=Config.device)
#     y = model(x, noise_level)
#     print(f"Here are some model outputs on random noise:")
#     show(y.denoised.clamp(0, 1))
#     model.train()
    
# demo_model(model)

# %% [markdown]
# # Train Model

# %%
from trainloader import Trainer, generate_images

trainer = Trainer(model, avg_model, d_train)

# %%
trainer.train(n_seconds=2*60*60)

# %%
# @torch.no_grad()
# def generate_images(model, n_images=16, n_steps=100, step_size=2.0):
#     model.eval()
#     x, prev = torch.rand(n_images, *Config.shape, device=Config.device), None
#     noise_levels = torch.linspace(1, 0, n_steps + 1, device=Config.device)
#     for nl_in, nl_out in zip(noise_levels, noise_levels[1:]):
#         denoised = pred = model(x, nl_in.view(1, 1, 1, 1)).denoised
#         if prev is not None: denoised = prev + step_size * (denoised - prev)
#         x, prev = alpha_blend(x, denoised, nl_out / nl_in), pred
#     model.train()
#     return x.clamp(0, 1)


# def demo_sample_grids(dataset, model, rows=8, cols=8):
#     real_rows, fake_rows = [], []
#     for i in tqdm(range(rows)):
#         real_rows.append(torch.cat([random.choice(dataset).im for _ in range(cols)], -1))
#         fake_rows.append(torch.cat(tuple(generate_images(model, n_images=cols)), -1))
#     real_im = torch.cat(real_rows, -2)
#     padding = torch.ones_like(real_im[..., :32])
#     fake_im = torch.cat(fake_rows, -2).cpu()
#     return TF.to_pil_image(torch.cat([real_im, padding, fake_im], -1))
# demo_sample_grids(d_train, avg_model)

# %%
torch.save(model.state_dict(), 'model5.pth')


