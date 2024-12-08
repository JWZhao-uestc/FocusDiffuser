from torch.autograd import Variable
import enum
import torch.nn.functional as F
from torchvision.utils import save_image
import torch
import math
# from visdom import Visdom
# viz = Visdom(port=8850)
import numpy as np
import torch as th
#from .train_util import visualize
#from .nn import mean_flat
#from .losses import normal_kl, discretized_gaussian_log_likelihood
from scipy import ndimage
from torchvision import transforms
import sys
import argparse
import cv2




def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()

    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


##########
class Sample():
    def __init__(self, noise_schedule='linear', steps=1000):
        super().__init__()
        self.noise_schedule=noise_schedule
        self.betas = get_named_beta_schedule(noise_schedule, steps)
        self.betas = np.array(self.betas, dtype=np.float64)
        #betas = betas
        assert len(self.betas.shape) == 1, "betas must be 1-D"
        assert (self.betas > 0).all() and (self.betas <= 1).all()
        self.num_timesteps = int(self.betas.shape[0])

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod - 1)


    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        # print('sqrt_alphas_cumprod', sqrt_alphas_cumprod.shape)
        # print('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod.shape)
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def ddim_sample(self, x_t, t, x_start, eta=1.):

        pred_noise = self.predict_noise_from_start(x_t, t, x_start)
        alpha = self.alphas_cumprod[t]
        time_next = t-1
        alpha_next = self.alphas_cumprod[time_next]

        sigma = eta * np.sqrt((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha))
        c = np.sqrt(1 - alpha_next - sigma ** 2)

        noise = torch.randn_like(x_start)

        x_t_next = x_start * np.sqrt(alpha_next) + \
                c * pred_noise + \
                sigma * noise
        return x_t_next


def add_noise_to_mask(x_start, t, sampler):
    #noise = th.randn_like(x_start)
    res_t = sampler.q_sample(x_start, t)
    return res_t
    
def Train_timestamp_random_sample(all_time_steps, batch_size):
    import numpy as np
    w = np.ones([all_time_steps])
    p = w / np.sum(w)
    indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
    indices = th.from_numpy(indices_np).long()
    weights_np = 1 / (len(p) * p[indices_np])
    return indices.cuda(), weights_np
    #print(weights_np)


from tqdm import tqdm
import os
if __name__ == "__main__":
 
    img_path = '../../DATA/train/GT/camourflage_00004.png'
    #'../../DATA/train/GT/COD10K-CAM-4-Amphibian-68-Toad-4954.png'
    image = cv2.resize(cv2.imread(img_path, 0), (256,256))
    image = torch.Tensor(image) // 255
    image = (image*2 - 1)*0.1
    all_steps = 1000
    Diffusion_images_path = '../../Diffusion_images'
    os.makedirs(Diffusion_images_path, exist_ok=True)

    shape = image[None,:,:].shape
    sampler = Sample(noise_schedule = 'linear', steps = all_steps)
    for i in tqdm(range(all_steps)):
        t = th.tensor([i]*shape[0])
        #print(t)
        res_t = add_noise_to_mask(image[None,:,:], t, sampler)
        res_t = torch.clamp(res_t, min=-1*0.1, max=1*0.1) ###########
        res_t = ((res_t / 0.1) + 1) / 2.

        # pred_noise = sampler.predict_noise_from_start(res_t, t, x0)
        # print(pred_noise)

        res = res_t.squeeze().cpu().detach().numpy()*255
        dir_path = os.path.join(Diffusion_images_path, img_path.split('/')[-1].split('.')[0])
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, str(i)+'.png')
        cv2.imwrite(path, res)
    
    # # sample t while training 
    # t, weight = Train_timestamp_random_sample(all_steps, batch_size=6)
    # print(t)
    # for i in tqdm(t):
    #     t = th.tensor([i]*shape[0])
    #     #print(t)
    #     res_t = add_noise_to_mask(image[None,:,:], t, all_steps)
    #     res = res_t.squeeze().cpu().detach().numpy()*255
    #     cv2.imwrite(str(i)+'.png', res)
    # #weights = th.from_numpy(weights_np).float().to(device)



    
    
