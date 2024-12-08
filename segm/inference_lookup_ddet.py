from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as Ft
import torchvision.transforms.functional as F
import argparse
import os
import segm.utils.torch as ptu
from segm.data import transform
from segm.data.utils import STATS
from segm.data.ade20k import ADE20K_CATS_PATH
from segm.data.utils import dataset_cat_description, seg_to_rgb
import matplotlib.pyplot as plt
from segm.model.factory import load_model, load_diff_ddet_model
from segm.model.utils import inference
import cv2
import glob
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from torchstat import stat
from thop import profile
from thop import clever_format

def main(args):
  
    scale = args.scale
    sampling_timesteps = args.sampling_timesteps
    model_path = args.model_path
    input_dir =  args.input_dir 
    output_dir = args.output_dir + model_path.split('/')[-1].split('diff')[-1].replace('.pth','')
    db_name = args.db_name
    im_size = args.im_size
    ptu.set_gpu_mode(True)

    model_dir = Path(model_path).parent
    model, variant = load_diff_ddet_model(model_path, scale, sampling_timesteps)
    model.to(ptu.device)

    normalization_name = variant["dataset_kwargs"]["normalization"]
    normalization = STATS[normalization_name]

    trans = transform.Compose([
                transform.Resize((im_size, im_size), (288,288)),
                transform.ToTensor(),
                transform.Normalize(mean=normalization['mean'], std=normalization['std'])
            ])

    input_dir = Path(os.path.join(input_dir, db_name))
    output_dir = os.path.join(output_dir, db_name)
    os.makedirs(output_dir, exist_ok=True)

    list_dir = glob.glob(os.path.join(input_dir, 'Imgs', '*.jpg'))
    print(os.path.join(input_dir, 'Imgs', '.jpg'))
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()
    
    for filepath in tqdm(list_dir):

        if args.type == 'mask':
            gt_path = filepath.replace('Imgs', 'GT').replace('jpg', 'png')
        else:
            gt_path = filepath.replace('Imgs', 'Edge').replace('jpg', 'png')
        pil_im = cv2.imread(filepath, 1)
        plt_pil_im = plt.imread(filepath, 1)
        gt_np = cv2.imread(gt_path, 0)
        im_tensor, mask = trans(pil_im, gt_np)
        gt = mask.to(ptu.device).unsqueeze(0)
        mask = (mask.cpu().detach().numpy() > 0)*255
 
        im_tensor = im_tensor.to(ptu.device).unsqueeze(0)
        #pred_edge,

        with torch.no_grad():    
            logits, lookup_ori, pred_edge, lookup  = model.forward(im_tensor, gt)
           

        for t, logit in enumerate(logits):
            logit_softmax = logit#Ft.softmax(logit, 1)
            pred_for_compute = logit_softmax[0,0].cpu().detach().numpy()
            pred_for_compute =(pred_for_compute-pred_for_compute.min())/(pred_for_compute.max()-pred_for_compute.min()+1e-8)*255
            pred_edge_gt = pred_edge[0,0].cpu().detach().numpy()*255
            
            
            if pred_for_compute.shape[0] != gt_np.shape[0] or pred_for_compute.shape[1] != gt_np.shape[1]:
                pred_for_compute = cv2.resize(pred_for_compute, (gt_np.shape[1], gt_np.shape[0]))
                mask = gt_np
                pred_edge_gt = cv2.resize(pred_edge_gt, (gt_np.shape[1], gt_np.shape[0]))

            # #Write Image
            # output_path = os.path.join(output_dir_path, str(t+1)+'.png')
            # img1 = np.concatenate((pil_im, np.expand_dims(mask, 2).repeat(3, -1)), 1)
            # img2 = np.concatenate((img1, np.expand_dims(pred_for_compute, 2).repeat(3, -1)), 1)
            # img2 = np.concatenate((img2, np.expand_dims(pred_edge_gt, 2).repeat(3, -1)), 1)
            # #img2 = np.concatenate((img2, np.expand_dims(lookup_pred_edge_gt, 2).repeat(3, -1)), 1)
            # cv2.imwrite(output_path, img2)
    
        SM.step(pred=pred_for_compute, gt=mask)

        pred_for_compute[np.where(pred_for_compute < 100)] = 0
        pred_for_compute = (pred_for_compute > 128) * 255
        
        WFM.step(pred=pred_for_compute, gt=mask)
        FM.step(pred=pred_for_compute, gt=mask)
        EM.step(pred=pred_for_compute, gt=mask)
        M.step(pred=pred_for_compute, gt=mask)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = M.get_results()["mae"]

    results = {
        "Smeasure": sm,               # S-measure
        "wFmeasure": wfm,             # weighted-F-measure
        "meanFm": fm["curve"].mean(), # mean-F-measure
        "meanEm": em["curve"].mean(), # mean-E-measure
        "maxEm": em["curve"].max(),   # max-E-measure
        "MAE": mae,                   # MAE
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
    file=open(os.path.join(str(Path(output_dir).parent), db_name+'.txt'), "a")
    file.write(db_name+ ': ' + '#'*30 + '\n')
    for key in results.keys():
        file.write(key + ': '+str(results[key])+'\n')

    print("Eval finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--scale", default=2.0, type=float, help="signal scale")
    parser.add_argument("-s_t", "--sampling_timesteps", default= 4, type=int, help="ddim sample itrations")
    parser.add_argument("--model-path", type=str, default='../seg_tiny_1024/checkpoint.pth')
    parser.add_argument("--input-dir", "-i", default='../../data/DIS5K/DIS-VD/im',type=str, help="folder with input images")
    parser.add_argument("--output-dir", "-o", type=str, default='../res',help="folder with output images")
    parser.add_argument("--db_name", "-d", type=str,help="which dataset")
    parser.add_argument("--im-size", type=int, default=512,help="folder with output images")
    parser.add_argument("--type", type=str, default='mask',help="mask or edge")
    args = parser.parse_args()
    main(args)