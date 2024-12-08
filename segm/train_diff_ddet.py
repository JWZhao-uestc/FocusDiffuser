import sys
from pathlib import Path
import yaml
import json
import numpy as np
import torch
#import click
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP

#from segm.utils import distributed
import segm.utils.torch as ptu
from segm import config

from segm.model.factory import create_diff_ddet_segmenter
from segm.optim.factory import create_optimizer, create_scheduler
from segm.data.factory import create_dataset
from segm.model.utils import num_params

from timm.utils import NativeScaler
from contextlib import suppress

#from segm.utils.distributed import sync_model
from segm.engine import train_one_epoch, evaluate

def main(args):
    # # start distributed mode
    ptu.set_gpu_mode(True)
    # distributed.init_process()
    scale = args.scale
    sampling_timesteps = args.sampling_timesteps
    freeze_condition_embed = args.freeze_condition_embed
    freeze_post_process = args.freeze_post_process
    data_root = args.data_root
    db_name = args.db_name
    log_dir = args.log_dir
    dataset = args.dataset
    im_size = args.im_size
    crop_size = args.crop_size
    window_size = args.window_size
    window_stride = args.window_stride
    backbone = args.backbone
    decoder = args.decoder
    optimizer = args.optimizer
    scheduler = args.scheduler
    weight_decay = args.weight_decay
    dropout = args.dropout
    drop_path = args.drop_path
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    normalization = args.normalization
    eval_freq = args.eval_freq
    amp = args.amp
    resume = args.resume
    min_lr = args.min_lr
 
    # set up configuration
    cfg = config.load_config()
    model_cfg = cfg["model"][backbone]
    dataset_cfg = cfg["dataset"][dataset]
    if "mask_transformer" in decoder:
        decoder_cfg = cfg["decoder"]["mask_transformer"]
    else:
        decoder_cfg = cfg["decoder"][decoder]

    # model config
    if not im_size:
        im_size = dataset_cfg["im_size"]
    if not crop_size:
        crop_size = dataset_cfg.get("crop_size", im_size)
    if not window_size:
        window_size = dataset_cfg.get("window_size", im_size)
    if not window_stride:
        window_stride = dataset_cfg.get("window_stride", im_size)

    model_cfg["image_size"] = (crop_size, crop_size)
    model_cfg["backbone"] = backbone
    model_cfg["dropout"] = dropout
    model_cfg["drop_path_rate"] = drop_path
    decoder_cfg["name"] = decoder
    model_cfg["decoder"] = decoder_cfg

    # dataset config
    world_batch_size = dataset_cfg["batch_size"]
    num_epochs = dataset_cfg["epochs"]
    lr = dataset_cfg["learning_rate"]
    if batch_size:
        world_batch_size = batch_size
    if epochs:
        num_epochs = epochs
    if learning_rate:
        lr = learning_rate
    if eval_freq is None:
        eval_freq = dataset_cfg.get("eval_freq", 1)

    if normalization:
        model_cfg["normalization"] = normalization

    # experiment config
    batch_size = world_batch_size // ptu.world_size
    variant = dict(
        world_batch_size=world_batch_size,
        version="normal",
        resume=resume,
        dataset_kwargs=dict(
            data_root = data_root,
            dataset=dataset,
            image_size=im_size,
            crop_size=crop_size,
            batch_size=batch_size,
            normalization=model_cfg["normalization"],
            split="train",
            num_workers=4,
        ),
        algorithm_kwargs=dict(
            batch_size=batch_size,
            start_epoch=0,
            num_epochs=num_epochs,
            eval_freq=eval_freq,
        ),
        optimizer_kwargs=dict(
            opt=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
            clip_grad=None,
            sched=scheduler,
            epochs=num_epochs,
            min_lr=min_lr,
            poly_power=0.9,
            poly_step_size=1,
        ),
        net_kwargs=model_cfg,
        amp=amp,
        log_dir=log_dir,
        inference_kwargs=dict(
            im_size=im_size,
            window_size=window_size,
            window_stride=window_stride,
        ),
    )

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    encoder_checkpoint_path = log_dir / "checkpoint_cod_aug.pth"
    checkpoint_path = log_dir / "checkpoint_cod_diff_stage1_scale01_add_RFB_iou_loss_pvt_concat4layers_noise288_ex4_2_2_radius2_train_backbone_COD10K_1000_offline_augx2.pth"#"checkpoint_cod_aug_pvtv2_v5_add_edge.pth"
    assert checkpoint_path.exists(), 'please check your checkpoint!!!'
    # dataset
    dataset_kwargs = variant["dataset_kwargs"]

    train_loader = create_dataset(dataset_kwargs)
    val_kwargs = dataset_kwargs.copy()
    val_kwargs["split"] = "val"
    val_kwargs["batch_size"] = 1
    val_kwargs["crop"] = False
    val_loader = create_dataset(val_kwargs)
    n_cls = train_loader.unwrapped.n_cls

    # model
    net_kwargs = variant["net_kwargs"]
    net_kwargs["n_cls"] = n_cls
    net_kwargs["scale"] = scale
    net_kwargs["sampling_timesteps"] = sampling_timesteps

    model = create_diff_ddet_segmenter(net_kwargs)
    #print('model', model)
    model.to(ptu.device)

    # optimizer
    optimizer_kwargs = variant["optimizer_kwargs"]
    optimizer_kwargs["iter_max"] = len(train_loader) * optimizer_kwargs["epochs"]
    optimizer_kwargs["iter_warmup"] = 0.0
    opt_args = argparse.Namespace()
    opt_vars = vars(opt_args)
    for k, v in optimizer_kwargs.items():
        opt_vars[k] = v
    optimizer = create_optimizer(opt_args, model)
    # lr_scheduler = create_scheduler(opt_args, optimizer)
    num_iterations = 0
    amp_autocast = suppress
    loss_scaler = None
    if amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    # resume
    if resume and checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_name = Path(checkpoint_path)
        print("Direct resume traing!!!!!!")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)  
        variant["algorithm_kwargs"]["start_epoch"] = 0

    if ptu.distributed:
        model = DDP(model, device_ids=[ptu.device], find_unused_parameters=True)

    # save config
    variant_str = yaml.dump(variant)
    print(f"Configuration:\n{variant_str}")
    variant["net_kwargs"] = net_kwargs
    variant["dataset_kwargs"] = dataset_kwargs
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "variant.yml", "w") as f:
        f.write(variant_str)

    # train
    start_epoch = variant["algorithm_kwargs"]["start_epoch"]
    num_epochs = variant["algorithm_kwargs"]["num_epochs"]
    eval_freq = variant["algorithm_kwargs"]["eval_freq"]

    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module

    val_seg_gt = val_loader.dataset.get_gt_seg_maps()

    print(f"Train dataset length: {len(train_loader.dataset)}")
    print(f"Val dataset length: {len(val_loader.dataset)}")
    #print(f"Encoder parameters: {num_params(model_without_ddp.encoder)}")
    #print(f"Decoder parameters: {num_params(model_without_ddp.decoder)}")
    dice_max = 0.0 
    wfm_max = 0.0
    mae_min = 1000
    for epoch in range(start_epoch, num_epochs):
        # train for one epoch
        train_logger = train_one_epoch(
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            epoch,
            amp_autocast,
            loss_scaler,
        )
        # save checkpoint
        snapshot = dict(
            model=model_without_ddp.state_dict(),
            optimizer=optimizer.state_dict(),
            n_cls=model_without_ddp.n_cls,
            lr_scheduler=lr_scheduler.state_dict(),
        )
        if loss_scaler is not None:
            snapshot["loss_scaler"] = loss_scaler.state_dict()
        snapshot["epoch"] = epoch

        # save_checkpoint_path = log_dir / "checkpoint_cod_diff_c_edge_from_ddet.pth"
        # print('save_checkpoint_path', str(save_checkpoint_path))
        # torch.save(snapshot, save_checkpoint_path)
        
        # evaluate
        eval_epoch = epoch % eval_freq == 0 or epoch == num_epochs - 1
        if eval_epoch:
            dice, wfm, mae = evaluate(
                model,
                val_loader,
                val_seg_gt,
                window_size,
                window_stride,
                amp_autocast,
            )
        name = "trained_checkpoints.pth"
        save_checkpoint_path = log_dir / name
        if dice > dice_max or wfm > wfm_max or mae < mae_min:
            print('save_checkpoint_path', str(save_checkpoint_path))
            torch.save(snapshot, save_checkpoint_path)
            if dice > dice_max:
                dice_max = dice
            if wfm > wfm_max:
                wfm_max = wfm
            if mae < mae_min:
                mae_min = mae
    #distributed.barrier()
    #distributed.destroy_process()
    sys.exit(1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--freeze_condition_embed", action="store_true")
    parser.add_argument("--freeze_post_process", action="store_true")
    parser.add_argument("--scale", default=2.0, type=float, help="signal scale")
    parser.add_argument("-s_t", "--sampling_timesteps", default= 4, type=int, help="ddim sample itrations")
    parser.add_argument("--log-dir", type=str, default='seg_patch16_384_bce',help="logging directory")
    parser.add_argument("--data_root", type=str, default='./Raw_data', help="logging directory")
    parser.add_argument("--db_name", type=str, default='all', help="which dataset to train")
    parser.add_argument("--dataset", type=str, default='dis5k')
    parser.add_argument("--im-size", default=None, type=int, help="dataset resize size")
    parser.add_argument("--crop-size", default=None, type=int)
    parser.add_argument("--window-size", default=None, type=int)
    parser.add_argument("--window-stride", default=None, type=int)
    parser.add_argument("--backbone", default="vit_small_patch16_384", type=str)
    parser.add_argument("--decoder", default="mask_transformer", type=str)
    parser.add_argument("--optimizer", default="sgd", type=str)
    parser.add_argument("--scheduler", default="polynomial", type=str)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--drop-path", default=0.1, type=float)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--epochs", default=512, type=int)
    parser.add_argument("-lr", "--learning-rate", default=None, type=float)
    parser.add_argument("--min-lr", default=1e-6, type=float)
    parser.add_argument("--normalization", default=None, type=str)
    parser.add_argument("--eval-freq", default=1, type=int)
    parser.add_argument("--amp", default=False, type=bool)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n_gpus", default=False, type=bool)
    args = parser.parse_args()
    main(args)
