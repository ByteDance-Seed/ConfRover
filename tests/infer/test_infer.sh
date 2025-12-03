#!/bin/bash

WANDB_PROJ=ConfRover_test # dump to this test tracking project
USER=test_user

# ************************************************************************
#   Preset arguments
# ************************************************************************

CKPT_PATH=/mnt/hdfs/gdd/projects/confrover/users/yuning/20250806_confrover1_release/model_weights/confrover_base/confrover_base.pt
OUTPUT_DIR=./test_output
TASK_NAME=test_infer

args=(
    experiment=generate
    task_name=$TASK_NAME
    paths.log_dir=$OUTPUT_DIR
    ckpt_path=$CKPT_PATH
    paths=hdfs
    #### Data ####
    +data/gen_dataset=atlasTest_fwd_varyStart
    ### Fast test through train/val/val_gen ####
    trainer.limit_predict_batches=1
    # trainer.fast_dev_run=true
    model.decoder.sampler.diffusion_steps=5
)

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
bash scripts/generate.sh "${args[@]}"
# bash scripts/debug_infer.sh "${args[@]}"


