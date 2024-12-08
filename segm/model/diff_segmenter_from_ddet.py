# (ex4_2_radius=2_add_and_add)
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion_func import add_noise_to_mask, Train_timestamp_random_sample, Sample
from segm.model.utils import padding, unpadding, LayerNorm2D
from segm.model.corr import CorrBlock
from timm.models.layers import trunc_normal_
from .unet import UNetModel
import cv2
import math
import numpy as np
from collections import namedtuple
from tqdm import tqdm
from skimage import segmentation
from .nn import zero_module
  

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU()
        self.branch0 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1))
        self.branch1 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1), BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)), BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)), BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3))
        self.branch2 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1), BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)), BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)), BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5))
        self.branch3 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1), BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)), BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)), BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7))
        self.conv_cat = BasicConv2d((4 * out_channel), out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.relu((x_cat + self.conv_res(x)))
        return x
    
class Conv_Block(nn.Module):
    def __init__(self, channels):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(channels*4, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels*2)

        self.conv3 = nn.Conv2d(channels*2, channels*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels*4)

        self.out_embed_ch = channels*4

    def forward(self, input1, input2, input3, input4):
        fuse = torch.cat((input1, input2, input3, input4), 1)
        fuse = self.bn1(self.conv1(fuse))
        fuse = self.bn2(self.conv2(fuse))
        fuse = self.bn3(self.conv3(fuse))
        return fuse

    def initialize(self):
        weight_init(self)

class DDetDiffSegmenter(nn.Module):
    def __init__(
        self,
        encoder,
        #decoder,
        n_cls,
        scale,
        sampling_timesteps,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = 8 #encoder.patch_size
        self.up_patch_size = 6 # 7:336
        self.encoder = encoder
        #self.decoder = decoder

        #add for pvt

        channels = 128
        self.side_conv1 = nn.Conv2d(512, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv2 = nn.Conv2d(320, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv3 = nn.Conv2d(128, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv4 = nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1)

        self.conv_block = Conv_Block(channels)


        self.condition_dim = 16
        self.image_size= 48*self.up_patch_size #128                                   #self.patch_size // (512 // self.image_size)
        self.decoder_embed = nn.Linear(self.conv_block.out_embed_ch, (self.up_patch_size) ** 2 * self.condition_dim, bias=True)
        self.RFB = RFB_modified(self.condition_dim, self.condition_dim)
        #self.RFB2 = RFB_modified(3, self.condition_dim)
        self.rescale_timesteps = True
        #self.num_timesteps = 1000
        self.decoder_embed_dim = 16                                     #self.patch_size // (512 // self.image_size)
        self.decoder_embed_edge = nn.Linear(self.conv_block.out_embed_ch, (self.up_patch_size) ** 2 * self.condition_dim, bias=True)
        self.conv_edge = nn.Sequential(
                nn.Conv2d(self.decoder_embed_dim, self.decoder_embed_dim, kernel_size=1, padding=0, ),
                nn.BatchNorm2d(self.decoder_embed_dim),
                nn.GELU(),
                nn.Conv2d(self.decoder_embed_dim, self.decoder_embed_dim, kernel_size=3, padding=1, ),
                nn.BatchNorm2d(self.decoder_embed_dim),
                nn.GELU(),
        )         
        
        self.final_lookup_downsample = nn.Sequential(
            nn.Conv2d(4*5*5, 128, kernel_size=3, stride=2, padding=1), #####radius:2: 4*5*5
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            zero_module(nn.Conv2d(512, 512, kernel_size=1, padding=0))
        )
         
        self.concat_lookup_and_edfeature = nn.Sequential(
                #nn.Conv2d(2*self.decoder_embed_dim, self.decoder_embed_dim, kernel_size=3, padding=1, ),
                nn.Conv2d(self.decoder_embed_dim, self.decoder_embed_dim, kernel_size=3, padding=1, ),
                nn.BatchNorm2d(self.decoder_embed_dim),
                nn.GELU(),
                nn.Conv2d(self.decoder_embed_dim, self.decoder_embed_dim, kernel_size=3, padding=1, ),
                nn.BatchNorm2d(self.decoder_embed_dim),
                nn.GELU(),
                nn.Conv2d(self.decoder_embed_dim, self.decoder_embed_dim, kernel_size=3, padding=1, ),
                nn.BatchNorm2d(self.decoder_embed_dim),
                nn.GELU(),
        ) 
        self.decoder_pred_edge = nn.Sequential(
                nn.Conv2d(self.decoder_embed_dim, self.decoder_embed_dim, kernel_size=3, padding=1, ),
                nn.BatchNorm2d(self.decoder_embed_dim),
                nn.GELU(),
                nn.Conv2d(self.decoder_embed_dim, 1, kernel_size=1, bias=True),
                nn.Sigmoid()
        )
        self.edge_lookup_downsample = nn.Sequential(
                nn.Conv2d(self.decoder_embed_dim, self.decoder_embed_dim*4, kernel_size=3, stride=2, padding=1, ),
                nn.BatchNorm2d(self.decoder_embed_dim*4),
                nn.GELU(),
                nn.Conv2d(self.decoder_embed_dim*4, self.decoder_embed_dim*4, kernel_size=3, stride=2, padding=1, ),
                nn.BatchNorm2d(self.decoder_embed_dim*4),
                nn.GELU(),
                nn.Conv2d(self.decoder_embed_dim*4, self.decoder_embed_dim*8, kernel_size=3, stride=2, padding=1, ),
                nn.BatchNorm2d(self.decoder_embed_dim*8),
                nn.GELU(),
                nn.Conv2d(self.decoder_embed_dim*8, self.decoder_embed_dim*8, kernel_size=1, padding=0, ),
                nn.BatchNorm2d(self.decoder_embed_dim*8),
                nn.GELU(),
        )

        #self.sampler = Sample(noise_schedule = 'linear', steps = self.num_timesteps)
        attention_resolutions="16,8"
        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(self.image_size // int(res))
        #self.attention_ds = attention_ds
        self.diff = UNetModel(
            image_size=self.image_size,
            in_channels=self.condition_dim+1, #condition and noise
            model_channels=128,
            out_channels=1,#(3 if not learn_sigma else 6),
            num_res_blocks=2,
            attention_resolutions=tuple(attention_ds),
            dropout=0.0,
            channel_mult=(1, 1, 2, 3, 4), #(1, 1, 2, 2, 4, 4),
            num_classes=None, #(NUM_CLASSES if class_cond else None),
            use_checkpoint=False,#True,#False, !!!!!!!!!!!!!!!save memory but cost time
            use_fp16=False,
            num_heads=4,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=False,
            use_new_attention_order=False,
        )

        # build Diffusion
        timesteps = 1000
        sampling_timesteps = sampling_timesteps#4 #cfg.MODEL.DiffusionDet.SAMPLE_STEP
        
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = scale #2.0#cfg.MODEL.DiffusionDet.SNR_SCALE
        self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def coords_grid(self, batch, ht, wd, device):
        coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
        coords = torch.stack(coords[::-1], dim=0).float()
        return coords[None].repeat(batch, 1, 1, 1)
    
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params
    
    def unpatchify(self, x, d_dim):
        """
        x: (N, L, patch_size**2 *d_dim)
        imgs: (N, 3, H, W)
        """
        p = self.up_patch_size #int(self.patch_size // (512 // self.image_size))
        if len(x.shape) == 3:
            w = int(x.shape[1]**.5)
            h = w
            assert h * w == x.shape[1]
        if len(x.shape) == 4:
            w = x.shape[2]
            h = x.shape[1]
            assert h == w
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, d_dim))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], d_dim, h * p, w * p))
        return imgs

    def condition_encoder(self, im):
        # 1.condition generation 
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)
        h, w = H // self.patch_size, W // self.patch_size
        
        features = self.encoder(im)
        x1 = features[0] #512
        x2 = features[1] #320
        x3 = features[2] #128
        x4 = features[3] #64

        # ####### multi scale
        E4, E3, E2, E1= self.side_conv1(x1), self.side_conv2(x2), self.side_conv3(x3), self.side_conv4(x4)

        if E4.size()[2:] != E2.size()[2:]:
            E4 = F.interpolate(E4, size=E2.size()[2:], mode='bilinear')
        if E3.size()[2:] != E2.size()[2:]:
            E3 = F.interpolate(E3, size=E2.size()[2:], mode='bilinear')
        if E1.size()[2:] != E2.size()[2:]:
            E1 = F.interpolate(E1, size=E2.size()[2:], mode='bilinear')

        E5 = self.conv_block(E4, E3, E2, E1)

        b, E5_c, E5_h, E5_w = E5.shape
        x = E5.view(b, E5_c, -1).permute(0,2,1)
        x_trans = x
        
        # reshape to [b, c, 128,128]
        b,N,c = x.shape
        x = self.decoder_embed(x) #[b,h*w,conditoin_dim*p**2]
        x_1 = self.unpatchify(x, d_dim=self.condition_dim)
        x = self.RFB(x_1)

        # edge_feature
        x_trans = self.decoder_embed_edge(x_trans)
        x_edge_conv = self.unpatchify(x_trans, self.decoder_embed_dim)
        edge_feature = self.conv_edge(x_edge_conv)
        
        # predict edge
        #edge_enhanced_feature_ = self.concat_lookup_and_edfeature(torch.cat((x_lookup, edge_feature), 1))
        edge_enhanced_feature_ = self.concat_lookup_and_edfeature(edge_feature)
        pred_edge = self.decoder_pred_edge(edge_enhanced_feature_)
        edge_enhanced_feature = edge_feature*pred_edge    ######## edge_feature   #[b,16,288,288]
        # look up
        edge_enhanced_feature = self.edge_lookup_downsample(edge_enhanced_feature) #[6, 128, 36, 36]
        x_lookup = edge_enhanced_feature#
        lookup_fn = CorrBlock(x_lookup, x_lookup, num_levels=4, radius=2) # num_levels=1 corr:4: 4
        coords = self.coords_grid(b,(self.image_size//self.patch_size),(self.image_size//self.patch_size),x_lookup.device)
        lookup_ori = lookup_fn(coords) #[b, 4x9x9, 32,32]
        lookup = self.final_lookup_downsample(lookup_ori)
        
        return x, edge_enhanced_feature, pred_edge, lookup
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def prepare_diffusion_concat(self, gt):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,)).long().cuda()
        noise = torch.randn_like(gt).cuda()
          
        x_start = gt
        x_start = (x_start * 2. - 1.) * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise) #########

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale) ###########
        x = ((x / self.scale) + 1) / 2.  ############

        diff_gt = x.float()

        return diff_gt, noise, t

    def prepare_targets(self, gt):
        diffused_gt, d_noise, d_t = self.prepare_diffusion_concat(gt)
        return gt, diffused_gt, d_noise, d_t
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, im, backbone_feats, pred_edge, lookup, input_noise, coarse_gt, t, infer_idx, x_self_cond=None, clip_x_start=False):
        
        #coarse_gt = input_noise
        input_noise = torch.clamp(input_noise, min=-1 * self.scale, max=self.scale)
        input_noise = ((input_noise / self.scale) + 1) / 2
        input_noise = input_noise.float() #######
        
        # generate coarse gt
        coarse_gt = (coarse_gt-coarse_gt.min())/(coarse_gt.max()-coarse_gt.min()+1e-8)*255
        proposal_img = self.get_proposal_img(im, coarse_gt, pred_edge, infer_idx) ############
        #cv2.imwrite('eval_proposal_img_'+str(infer_idx)+'.png', proposal_img[0,0].cpu().detach().numpy()*255)
        #proposal_img = self.RFB2(proposal_img)

        diff_input = torch.cat((backbone_feats, input_noise), 1)
        pred_gt = self.diff(diff_input, lookup, proposal_img, self._scale_timesteps(t))

        x_start = pred_gt  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(input_noise, t, x_start)

        pred_gt = F.sigmoid(pred_gt)
        pred_gt = torch.clamp(pred_gt, min=0.1, max=1)

        return ModelPrediction(pred_noise, x_start), pred_gt
    
    @torch.no_grad()
    def ddim_sample(self, im, backbone_feats, pred_edge, lookup, noise, clip_denoised=True, do_postprocess=True):
        batch = backbone_feats.shape[0]
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        #print(times)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        # print('time_pairs', time_pairs)
        # exit()
        img = torch.randn_like(noise).cuda()
        coarse_gt = torch.zeros_like(noise).cuda()
        x_start = None
        out_sets = []
        
        for infer_idx, (time, time_next) in enumerate(time_pairs):
            time = int(time)
            time_next = int(time_next)
            time_cond = torch.full((batch,), time).long().cuda()
            self_cond = x_start if self.self_condition else None
   
            preds, pred_gt = self.model_predictions(im, backbone_feats, pred_edge, lookup, img, coarse_gt, time_cond, infer_idx)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start
            #ori
            # x_start = F.sigmoid(x_start) # same with training !!!!!!!!!!!!!!!!!!!
            # coarse_gt = x_start
            # out_sets.append(x_start)

            x_start = F.sigmoid(x_start) # same with training !!!!!!!!!!!!!!!!!!!
            coarse_gt = x_start
            out_sets.append(pred_gt) #### train:x_start / val:pred_gt
            
            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
           
        return out_sets
    
    def get_proposal_bbox(self, gt, bias=10):
        """
        gt: b,1,h,w
        """
        batch = gt.shape[0]
        mask = np.zeros_like(gt.cpu().detach().numpy())
        for idx in range(batch):
            gt_idx = gt[idx][0]
            mask_sobel = np.zeros_like(gt_idx.cpu().detach().numpy())
            bd = segmentation.find_boundaries(gt_idx.cpu().detach().numpy(), mode="inner")
            mask_sobel[bd] = [255]
            if 255 in mask_sobel:
                #print(gt_idx.shape)
                sets = np.where(mask_sobel == 255)
                x_sets = sets[0]
                y_sets = sets[1]
                #print(y_sets.shape)
                y_lower = np.max(y_sets) #.cpu().detach().numpy()
                y_upper = np.min(y_sets)
                x_right = np.max(x_sets)
                x_left  = np.min(x_sets)
            
                x_r_bias = np.random.randint(-bias,bias)
                x_l_bias = np.random.randint(-bias,bias)
                y_u_bias = np.random.randint(-bias,bias)
                y_l_bias = np.random.randint(-bias,bias)
            
                x_left1  = max(x_left-x_l_bias, 0)
                x_right1 = min(x_right+x_r_bias, gt.shape[3])
                y_upper1 = max(y_upper-y_u_bias, 0)
                y_lower1 = min(y_lower+y_l_bias, gt.shape[2])
                #print(x_left, x_right, y_upper, y_lower)
                mask[idx,:,x_left1:x_right1, y_upper1:y_lower1] = 1

        #print(mask.shape)
        #cv2.imwrite('proposal_mask.png', mask[0,0]*255)
        #cv2.imwrite('mask.png', gt[0,0].cpu().detach().numpy()*255)
        
        return mask


    def get_proposal_img(self, im, gt, pred_edge, infer_idx=0):
        
        im = F.interpolate(im, (gt.shape[2], gt.shape[3]), mode='bilinear')
        # early concat_edge
        #im = torch.cat((im, pred_edge), 1) # [b, 3+1, h, w]
        
        if self.training: #train
            mask = self.get_proposal_bbox(gt)
            mask = torch.as_tensor(mask, device = im.device)
            proposal_im = im*mask
        else: # test
            if infer_idx == 0:
                mask = torch.zeros_like(gt)
                #cv2.imwrite('eval_proposal_mask'+str(infer_idx)+'.png', mask[0,0].cpu().detach().numpy()*255)
            else:
                mask = self.get_proposal_bbox(gt)
                #cv2.imwrite('eval_proposal_mask'+str(infer_idx)+'.png', mask[0,0]*255)
                mask = torch.as_tensor(mask, device = im.device)
            proposal_im = im*mask
            #cv2.imwrite('eval_inference_gt'+str(infer_idx)+'.png', gt[0,0].cpu().detach().numpy()*255)
        # late concat_edge
        proposal_im = torch.cat((proposal_im, pred_edge), 1) # [b, 3+1, h, w]
        
        return proposal_im

    def forward(self, im, gt):
        #gt = torch.ones((1,288,288))
        if not self.training:
            with torch.no_grad():
                gt = gt.unsqueeze(1)
                features, edge_con, pred_edge, lookup = self.condition_encoder(im)
                init_noise = torch.randn_like(gt).cuda() #input a pure noise
                results = self.ddim_sample(im, features, pred_edge, lookup, init_noise)
                #exit()
                #print(results)
            return results, edge_con, pred_edge, lookup

        if self.training:
            features, edge_con, pred_edge, lookup = self.condition_encoder(im)
            #print('features', features.shape)
            gt = gt.unsqueeze(1)
            targets, diffused_gt, noises, t = self.prepare_targets(gt)
            # print('targets', targets.shape)
            # print('diffused_gt', diffused_gt.shape)
            # print('noises', noises.shape)
            
            proposal_img = self.get_proposal_img(im, gt, pred_edge, infer_idx=0)
            # cv2.imwrite('img.png', im[0,0].cpu().detach().numpy()*255)
            # cv2.imwrite('proposal_img.png', proposal_img[0,0].cpu().detach().numpy()*255)
            
            diff_input = torch.cat((features, diffused_gt), 1)
            #print('diff_input', diff_input.shape)
            pred_gt = self.diff(diff_input, lookup, proposal_img, self._scale_timesteps(t))
            pred_gt = F.sigmoid(pred_gt) ############# SAME with line 256 !!!!!!!
            train_gt = gt 
            train_gt_pred = pred_gt 
        return train_gt, train_gt_pred, pred_edge #masks, 

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)
