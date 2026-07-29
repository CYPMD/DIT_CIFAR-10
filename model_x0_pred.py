#model_x0_pred.py
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange

from tqdm.auto import tqdm

from dit import DiT


def normalize_to_neg1_1(x):
    return x * 2 - 1

def unnormalize_to_0_1(x):
    return (x + 1) * 0.5

class RectifiedFlow(nn.Module):
    def __init__(
        self,
        net: DiT,
        device="cuda",
        channels=3,
        image_size=32,
        logit_normal_sampling_t=True,
    ):
        super().__init__()
        self.net = net
        self.device = device
        self.channels = channels
        self.image_size = image_size
        self.logit_normal_sampling_t = logit_normal_sampling_t
        self.t_eps = 5e-2

    def forward(self, x):
        if self.logit_normal_sampling_t:
            t = torch.randn((x.shape[0],), device=self.device).sigmoid()
        else:
            t = torch.rand((x.shape[0],), device=self.device)
        
        t_ = rearrange(t, "b -> b 1 1 1")
        z = torch.randn_like(x)
        x = normalize_to_neg1_1(x)
        z_t = (1 - t_) * x + t_ * z

        x_pred = self.net(z_t, t)

        # Simplified and numerically stable v-loss equivalent:
        denominator = t_.clamp_min(self.t_eps)
        loss = F.mse_loss(x_pred, x, reduction='none')
        weight = 1.0 / (denominator ** 2)
        
        return (loss * weight).mean()
    
    @torch.no_grad()
    def sample(self, batch_size, sample_steps=50, return_all_steps=False):
        z = torch.randn(
            (batch_size, self.channels, self.image_size, self.image_size),
            device=self.device,
        )
        
        images = [z]
        # Create intervals from 1.0 exactly down to 0.0
        t_span = torch.linspace(1.0, 0.0, sample_steps + 1, device=self.device)

        for i in tqdm(range(sample_steps)):
            t = t_span[i]
            t_next = t_span[i + 1]
            t_batch = t.repeat(batch_size)

            x_pred = self.net(z, t_batch)
            
            # Since t >= 1/sample_steps > 0, there is no division by zero here.
            # Unclamped division allows the final step to collapse perfectly to x_pred!
            v_t = (z - x_pred) / t

            # Correct Euler step
            z = z + (t_next - t) * v_t
            images.append(z)
        
        z = unnormalize_to_0_1(z.clip(-1, 1))
        
        if return_all_steps:
            return z, torch.stack(images)

        return z
