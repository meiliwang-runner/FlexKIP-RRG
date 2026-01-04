#!/bin/bash
export CUDA_VISIBLE_DEVICES=0\

# mimic_cxr
dataset="mimic_cxr"
# annotation="/media/wangyujie/CXPMRG_Bench_MambaXray_VL/daasets/mimic/mimic.json"
annotation="/media/wangyujie/CXPMRG_Bench_MambaXray_VL/daasets/mimic/mimic+inter+intra1.json"
base_dir="/media/wangyujie/CXPMRG_Bench_MambaXray_VL/daasets/mimic/image_224/"

# mimic_cxr
version="mimic0813_L_vf_mea_moe_pro"
savepath="save/$dataset/$version"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

python -u train_downstream_moe_promot.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 2  \
    --val_batch_size 4 \
    --vision_model /media/wangyujie/CXPMRG_Bench_MambaXray_VL/checkpoint/MambaXrayCLIP-L.pth \
    --freeze_vm True \
    --savedmodel_path ${savepath} \
    --max_length 80 \
    --min_new_tokens 60 \
    --max_new_tokens 100 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 4 \
    --learning_rate 1e-5 \
    --devices 1 \
    --max_epochs 10 \
    --limit_val_batches 1.0 \
    --val_check_interval 0.25 \
    --num_sanity_val_steps 2 \
    --precision 16-mixed \
    --do_sample True \
    --strategy auto \
    --temperature 0.7 \
    --use_sip True \
    --region_root /media/wangyujie/CXPMRG_Bench_MambaXray_VL/promot/region_top5_capture/feature_boxes_flat \
    --patch_size 40 \
    2>&1 |tee -a ${savepath}/log.txt

    