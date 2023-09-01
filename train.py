import os
import copy
import argparse

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data import load_transformed_data
from src.modules import UNet, EMA
from src.utils import save_images


class DDPM:
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=256,
        device="cuda"
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(
            1 - self.alpha_hat[t]
        )[:, None, None, None]
        return sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in range(self.noise_steps-1, 0, -1):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (1 / torch.sqrt(alpha)) * (((x - (1 - alpha)) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + beta * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(opt):
    device = opt.device
    data = load_transformed_data(opt.img_size, opt.batch_size)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)
    mse = nn.MSELoss()
    diffusion = DDPM(img_size=opt.img_size, device=device)
    logger = SummaryWriter()
    dlength = len(data)
    ema = EMA(0.9999)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(opt.epochs):
        pbar = tqdm(data)
        for i, (imgs, _) in enumerate(pbar):
            imgs = imgs.to(device)
            t = diffusion.sample_timesteps(imgs.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(imgs, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * dlength + i)

        if epoch % opt.save_every == 0:
            sampled_images = diffusion.sample(model, n=imgs.shape[0])
            img_path = os.path.join("results", f"{epoch}.jpg")
            save_images(sampled_images, img_path)
            ema_sampled_images = diffusion.sample(ema_model, n=imgs.shape[0])
            ema_img_path = os.path.join("results", f"ema_{epoch}.jpg")
            save_images(ema_sampled_images, ema_img_path)

            checkpoint_path = os.path.join("models", "ckpt.pt")
            torch.save(model.state_dict(), checkpoint_path)
            ema_checkpoint_path = os.path.join("models", "ema_ckpt.pt")
            torch.save(ema_model.state_dict(), ema_checkpoint_path)


def get_conf():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 500
    args.batch_size = 8
    args.img_size = 64
    args.device = "cuda"
    args.lr = 2e-4
    args.save_every = 1

    return args


if __name__ == "__main__":
    conf = get_conf()
    train(conf)
