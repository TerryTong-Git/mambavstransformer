import torch
import os
import datetime
import time
# from IPython.display import clear_output
from pathlib import Path
from collections import namedtuple
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from tqdm import tqdm

Sample = namedtuple("Sample", ("im", "noisy_im", "noise_level"))
from accelerate import Accelerator
accelerator = Accelerator()

device = accelerator.device
# def show(x):
#     if not isinstance(x, torch.Tensor) or x.ndim == 4:
#         x = torch.cat(tuple(x), -1)
#     display(TF.to_pil_image(x))

class Config:
    device = accelerator.device
    channels = 3
    image_size = 256
    shape = (channels, image_size, image_size)
    dataset = "imagenet-1k"

def alpha_blend(a, b, alpha):
    return alpha * a + (1 - alpha) * b


def to_device(ims):
    return Sample(*(x.to(Config.device) for x in ims))

@torch.no_grad()
def generate_images(model, n_images=16, n_steps=100, step_size=2.0):
    model.eval()
    x, prev = torch.rand(n_images, *Config.shape, device=Config.device), None
    noise_levels = torch.linspace(1, 0, n_steps + 1, device=Config.device)
    for nl_in, nl_out in zip(noise_levels, noise_levels[1:]):
        denoised = pred = model(x, nl_in.view(1, 1, 1, 1)).denoised
        if prev is not None: denoised = prev + step_size * (denoised - prev)
        x, prev = alpha_blend(x, denoised, nl_out / nl_in), pred
    model.train()
    return x.clamp(0, 1)



class Visualizer:
    def __init__(self):
        self.smoothed_loss = None
        self.losses_since_last_vis = []
        self.avg_losses = []
        self.steps = []
        self.step = 0
        self.t_last_vis = 0
        self.t_last_save = 0
        self.t_start = None
        folder, idx = datetime.datetime.now().strftime("%Y_%m_%d") + "_training_logs", 0
        while Path(f"{folder}_{idx}").exists():
            idx += 1
        self.folder = Path(f"{folder}_{idx}")
        self.folder.mkdir()
    def __call__(self, model, t, x, y, loss, n_demo=16):
        self.losses_since_last_vis.append(loss)
        self.smoothed_loss = loss if self.smoothed_loss is None else 0.99 * self.smoothed_loss + 0.01 * loss
        self.step += 1
        if self.t_start is None:
            self.t_start = t
        if t > self.t_last_vis + 30:
            generated_images = generate_images(model, n_images=n_demo)
            # clear_output(wait=True)
            # print("Input Noisified Image, Noise Level")
            # show(x.noisy_im[:n_demo])
            # show(x.noise_level.expand(len(x.noise_level), 3, 16, Config.image_size)[:n_demo])
            # print("Predictions")
            # show(y.denoised[:n_demo].clamp(0, 1))
            # print("Targets")
            # show(x.im[:n_demo])
            self.steps.append(self.step)
            self.avg_losses.append(sum(self.losses_since_last_vis) / len(self.losses_since_last_vis))
            self.losses_since_last_vis = []
            print("Generated Images (Averaged Model)")
            # show(generated_images)
            plt.title("Losses")
            plt.plot(self.steps, self.avg_losses)
            plt.gcf().set_size_inches(16, 4)
            plt.ylim(0, 1.5 * self.avg_losses[-1])
            if t > self.t_last_save + 120:
                torch.save(model.state_dict(), "model.pth")
                torch.save((self.steps, self.avg_losses), self.folder / "stats.pth")
                TF.to_pil_image(torch.cat(tuple(generated_images), -1)).save(self.folder / f"generated_{self.step:07d}.jpg", quality=95)
                plt.gcf().savefig(self.folder / "stats.jpg")
                self.t_last_save = t
            # plt.show()
            self.t_last_vis = t
        print(
            f"\r{self.step: 5d} Steps; {int(t - self.t_start): 3d} Seconds; "
            f"{60 * self.step / (t - self.t_start + 1):.1f} Steps / Min; "
            f"{len(x.im) * 60 * self.step / (t - self.t_start + 1):.1f} Images / Min; "
            f"Smoothed Loss {self.smoothed_loss:.5f}; "
        , end="")
        
class Looper(torch.utils.data.Dataset):
    def __init__(self, dataset, n=1<<20):
        self.dataset = dataset
        self.n = n
    def __len__(self):
        return max(len(self.dataset), self.n)
    def __getitem__(self, i):
        return self.dataset[i % len(self.dataset)]


class Trainer:
    def __init__(self, model, avg_model, dataset, batch_size=4):
        self.model = model
        self.avg_model = avg_model
        self.last_avg_time = time.time()
        self.opt = torch.optim.AdamW(model.parameters(), 3e-4, amsgrad=True)
        self.dataloader = torch.utils.data.DataLoader(Looper(dataset), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=64)
        self.dl_iter = iter(self.dataloader)
        self.visualizer = Visualizer()
        self.model, self.opt, self.dataloader,self.avg_model = accelerator.prepare(
        self.model, self.opt, self.dataloader,self.avg_model)

    def avg_model_step(self, t):
        if t > self.last_avg_time + 2:
            self.avg_model.update_parameters(self.model)
            self.last_avg_time = t

    def get_batch(self):
        try:
            batch = next(self.dl_iter)
        except StopIteration:
            self.dl_iter = iter(self.dataloader)
            batch = next(self.dl_iter)
        return to_device(batch)

    def train(self, n_seconds):
        # self.model.train()
        # start_time = time.time()
        # while time.time() < start_time + n_seconds:
        #     self.train_step(time.time())
        i=0
        for x in tqdm(self.dataloader):
            i+=1
            self.train_step(x,i)

    def train_step(self, x,t):
        # x = self.get_batch()
        y = self.model(x.noisy_im, x.noise_level)
        loss = F.mse_loss(y.denoised, x.im)
        self.opt.zero_grad(); accelerator.backward(loss); self.opt.step(); 
        if t%100==0:
            torch.save(self.model.state_dict(), "model.pth")

        # self.avg_model_step(t)
        # self.visualizer(self.avg_model, t, x, y, loss.item())

import os
import glob
import io
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_image(i, image_data_column, labels_column, output_dir, arrow_file_path):
    image_data = image_data_column[i]['bytes'].as_py()
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    image = Image.open(io.BytesIO(image_array))
    output_file_path = os.path.join(output_dir, f'{os.path.basename(arrow_file_path).split(".")[0]}_image_{i}.jpg')
    image.save(output_file_path)
    if labels_column:
        label = labels_column[i].as_py()
        with open(os.path.join(output_dir, f'{os.path.basename(arrow_file_path).split(".")[0]}_image_{i}.txt'), 'w') as label_file:
            label_file.write(str(label))

def process_arrow_file(arrow_file_path, output_dir):
    with open(arrow_file_path, 'rb') as f:
        reader = pa.ipc.open_stream(f)
        table = reader.read_all()

    image_data_column = table.column('image')
    labels_column = table.column('label') if 'label' in table.column_names else None

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, i, image_data_column, labels_column, output_dir, arrow_file_path) for i in range(len(image_data_column))]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {arrow_file_path}"):
            pass

def get_dataset(name):
    i = 0
    arrow_files_dir = name + "/train/"
    output_dir = './data'
    os.makedirs(output_dir, exist_ok=True)
    arrow_files = glob.glob(os.path.join(arrow_files_dir, '*.arrow'))
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_arrow_file, arrow_file_path, output_dir) for arrow_file_path in arrow_files]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing Arrow files"):
            pass

if __name__ == "__main__":
    get_dataset(Config.dataset)
