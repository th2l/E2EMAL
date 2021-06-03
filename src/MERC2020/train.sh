#!/bin/bash

trap "exit" INT

train_dir=./train_logs/sub_15_v1.1
video_frames_arg=16
audio_time_arg=4
mm_batch_size_arg=64

mm_steps_per_epoch=1.0
lr_init_val=5e-3
lr_min_val=1e-4
echo "Sleeping"
sleep 1
echo "Starting"

python -W ignore main.py --test run --video_frames $video_frames_arg --audio_time $audio_time_arg --feature_type mm --batch_size $mm_batch_size_arg --opt adam --lr_init $lr_init_val --min_lr $lr_min_val --use_min_lr 15 --epoch 20 --dir $train_dir --aag_audio 0.4 --ckpt $train_dir --steps_per_epoch $mm_steps_per_epoch
