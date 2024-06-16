#!/usr/bin/env bash
set -e
VK2PATH="/home/degbo/Desktop/SEDNet/datasets/vkitti/"
SAVEPATH="/home/degbo/Desktop/SEDNet/rs/"
SUFX=("morning")

# sednet epe<3std
LOADDIR="gwcnet-gc-elu-dropout-l1-logs-kl-3std-lr1e-4"
echo $LOADDIR
echo $CKPT
for S in ${SUFX[@]}
do
  echo $S
  python main.py --dataset vkitti2 \
    --datapath $VK2PATH --trainlist ./filenames/vkitti2_train.txt --testlist ./filenames/vkitti2_test_$S.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
    --batch_size 8 \
    --lr 0.0001 \
    --loss_type 'UC' \
    --save_test \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 3 \
    --model gwcnet-gcs \
    --logdir $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$S/ \
    --test_batch_size 1 \
    --device_id 0

  python post_process/generate_statistic.py --logdir $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$S/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 3 \
    --mask soft \
    --bin_scale log \
    --n_bins 11 \
    --uncert \
    --dataset vkitti2
done