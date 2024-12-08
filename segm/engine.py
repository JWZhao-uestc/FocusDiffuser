 # (ex4_2_backup)
import torch
import math
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
from segm.utils.logger import MetricLogger
from segm.metrics import gather_data, compute_metrics
from segm.model import utils
from segm.data.utils import IGNORE_LABEL
import segm.utils.torch as ptu
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    #print(list(range(1, len(tensor.shape))))
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def DICE_loss(pred, target):
    pred = pred.contiguous().view(pred.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()
    a = torch.sum(pred * target, 1)           # |X⋂Y|
    b = torch.sum(pred * pred, 1) + 0.001    # |X|
    c = torch.sum(target * target, 1) + 0.001  # |Y|
    d = (2 * a) / (b + c)
    return 1-d

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
 
    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
 
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def cal_ual(seg_logits, seg_gts):
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    #sigmoid_x = seg_logits.sigmoid()
    loss_map = 1 - (2 * seg_logits - 1).abs().pow(2)
    return loss_map.mean()

def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=11, stride=1, padding=5) - mask)
    #wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = F.binary_cross_entropy(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    #pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def IOU_Loss(pred, mask):
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()

def train_one_epoch(
    model,
    data_loader,
    optimizer,
    lr_scheduler,
    epoch,
    amp_autocast,
    loss_scaler,
):
    BCE_Loss = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    focal_Loss = FocalLoss()
    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 100

    model.train()
    data_loader.set_epoch(epoch)
    num_updates = epoch * len(data_loader)
    for batch in logger.log_every(data_loader, print_freq, header):
        im = batch["im"].to(ptu.device)
        #seg_gt = batch["segmentation"].long().to(ptu.device)
        seg_gt = batch["segmentation"].to(ptu.device)
        seg_edge_gt = batch["edge"].to(ptu.device)
        #print(seg_gt.shape)
        #exit()
        with amp_autocast():
            target, pred, pred_edge = model.forward(im, seg_gt)
            mse_loss = torch.mean(mean_flat((pred - target.detach())**2.))
            bce_loss = BCE_Loss(pred, target.detach()) # please check sigmoid !!!!!!
            iou_loss = IOU_Loss(pred, target.detach())
            edge_bce_loss = BCE_Loss(pred_edge, seg_edge_gt.unsqueeze(1))
            
            loss = mse_loss + 0.5*(bce_loss + iou_loss) + edge_bce_loss# + 0.5*focal_loss  # + 0.5*dice_loss
            #loss = bce_loss + iou_loss + edge_bce_loss # + 0.5*focal_loss  # + 0.5*dice_loss
            # loss = structure_loss(pred, target.detach())
            #print(loss.shape)
          
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
            )
        else:
            loss.backward()
            optimizer.step()

        num_updates += 1
        lr_scheduler.step_update(num_updates=num_updates)

        torch.cuda.synchronize()

        logger.update(
            loss=loss.item(),
            mse_loss=mse_loss.item(),
            bce_loss=bce_loss.item(),                            # please change the save checkpoint name !!!!!!!
            iou_loss = iou_loss.item(),                          # please change the save checkpoint name !!!!!!!
            #proposal_bce_loss=proposal_bce_loss.item(),          # please change the save checkpoint name !!!!!!!
            #proposal_iou_loss = proposal_iou_loss.item(),
            #proposal_edge_bce_loss=proposal_edge_bce_loss.item(),          # please change the save checkpoint name !!!!!!!
            edge_bce_loss = edge_bce_loss.item(),
            learning_rate=optimizer.param_groups[0]["lr"],
        )
        #break

    return logger

def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)
def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    val_seg_gt,
    window_size,
    window_stride,
    amp_autocast,
):
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module
    logger = MetricLogger(delimiter="  ")
    header = "Eval: COD10K" #"Eval: NC4K" # CAMO COD10K CHAMELEON
    print_freq = 200
    sam_dice_scores = []
    sam_iou_scores = []
    val_seg_pred = {}
    model.eval()
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()
    for batch in logger.log_every(data_loader, print_freq, header):
        im = batch["im"].to(ptu.device)
        seg_gt = batch["segmentation"].to(ptu.device)
        with amp_autocast():
            seg_pred, x_lookup, _ , _ = model.forward(im, seg_gt)

        for t, logit in enumerate(seg_pred):
            logit_softmax = logit #Ft.softmax(logit, 1)
            pred_for_compute = logit_softmax[0,0].cpu().detach().numpy()
            pred =(pred_for_compute-pred_for_compute.min())/(pred_for_compute.max()-pred_for_compute.min()+1e-8)*255

        mask = (seg_gt.squeeze().cpu().detach().numpy() > 0) * 255
    
        ##### all value
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        M.step(pred=pred, gt=mask)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = M.get_results()["mae"]

    results = {
        "Smeasure": sm,
        "wFmeasure": wfm,
        "meanFm": fm["curve"].mean(),
        "meanEm": em["curve"].mean(),
        "maxEm": em["curve"].max(),
        "MAE": mae,
        "adpEm": em["adp"],
        "adpFm": fm["adp"],
        "maxFm": fm["curve"].max(),
    }
    print('#'*30)
    print('Sm: %.4f' % sm)
    print('wF: %.4f' % wfm)
    print('mF: %.4f' % fm["curve"].mean())
    print('mE: %.4f' % em["curve"].mean())
    print('xE: %.4f' % em["curve"].max())
    print('M: %.4f' % mae)
    print(results)
    print('#'*30)
    
    # exit()
    return round(sm, 3), round(wfm, 3), round(mae, 3)       

