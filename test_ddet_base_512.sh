#CAMO
CUDA_VISIBLE_DEVICES=2 python -m segm.inference_lookup_ddet --scale 0.1 -s_t 4 --model-path ./seg_base_p16_384/checkpoint_cod_diff_stage1_scale01_add_RFB_iou_loss_pvt_concat4layers_noise288_ex4_2_2_radius2_train_backbone_val_CAMO.pth --input-dir ./DATA/test --output-dir ./Diff_Results --db_name CAMO --im-size 384
# #COD10K
#CUDA_VISIBLE_DEVICES=0 python -m segm.inference_ddet --scale 0.1 -s_t 4 --model-path ./seg_base_p16_384/checkpoint_cod_diff_stage1_scale01_add_RFB_iou_loss_pvt_concat4layers_noise288_ex4_2_2_radius2_train_backbone_COD10K_1000_offline_augx2_epoch5_8696.pth --input-dir ./DATA/test --output-dir ./Diff_Results --db_name COD10K --im-size 384
# #NC4K
#CUDA_VISIBLE_DEVICES=0 python -m segm.inference_ddet --scale 0.1 -s_t 4 --model-path ./seg_base_p16_384/checkpoint_cod_diff_stage1_scale01_add_RFB_iou_loss_pvt_concat4layers_noise288_ex4_2_2_radius2_train_backbone_COD10K_1000_offline_augx2_epoch5_8696.pth --input-dir ./DATA/test --output-dir ./Diff_Results --db_name NC4K --im-size 384
# #CHAMELEON
# CUDA_VISIBLE_DEVICES=0 python -m segm.inference_ddet --scale 0.1 -s_t 4 --model-path ./seg_base_p16_384/checkpoint_cod_diff_stage1_scale01_add_RFB_iou_loss_pvt_concat4layers_noise288_ex4_2_2_radius2_train_backbone_val_CAMO.pth --input-dir ./DATA/test --output-dir ./Diff_Results --db_name CHAMELEON --im-size 384