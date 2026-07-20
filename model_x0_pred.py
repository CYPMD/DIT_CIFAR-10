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

        # The network directly predicts the clean image.
        x_pred = self.net(z_t, t)

        # Convert the clean-image prediction and target to velocity space.
        denominator = t_.clamp_min(self.t_eps)
        v_t = (z_t - x_pred) / denominator
        target = (z_t - x) / denominator

        return F.mse_loss(target, v_t)
    
    @torch.no_grad()
    def sample(self, batch_size, sample_steps=50, return_all_steps=False):
        z = torch.randn(
            (
                batch_size,
                self.channels,
                self.image_size,
                self.image_size,
            ),
            device=self.device,
        )
        
        images = [z]
        t_span = torch.linspace(
            0,
            1,
            sample_steps,
            device=self.device,
        )

        for t in tqdm(reversed(t_span)):
            # Expand t from a 0D scalar to a 1D batch tensor.
            t_batch = t.repeat(batch_size)

            # The model output is a clean-image prediction.
            x_pred = self.net(z, t_batch)

            # Convert the clean-image prediction to velocity.
            v_t = (z - x_pred) / t.clamp_min(self.t_eps)

            z = z - v_t / sample_steps
            images.append(z)
        
        z = unnormalize_to_0_1(z.clip(-1, 1))
        
        if return_all_steps:
            return z, torch.stack(images)

        return z
