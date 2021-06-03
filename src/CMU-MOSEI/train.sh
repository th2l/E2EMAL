#!/bin/bash

trap "exit" INT

train_dir=./train_logs/E2EMLA_emo_sL_r8.5
video_frames_arg=16
audio_time_arg=4
mm_batch_size_arg=32

mm_steps_per_epoch=1.0
lr_init_val=1e-3
lr_min_val=1e-4
num_epochs=20
lr_min_start=15
echo "Sleeping"
sleep 1
echo "Starting"

#python -W ignore main.py --video_frames $video_frames_arg --audio_time $audio_time_arg --feature_type mm --batch_size $mm_batch_size_arg --opt adam --lr_init $lr_init_val --min_lr $lr_min_val --use_min_lr $lr_min_start --epoch $num_epochs --dir $train_dir --aag_audio 0.4 --ckpt $train_dir --steps_per_epoch $mm_steps_per_epoch
#
#echo "Sleeping"
#sleep 1
#echo "Starting"
#train_dir=./train_logs/E2EMLA_emo_r9
#python -W ignore main.py --video_frames $video_frames_arg --audio_time $audio_time_arg --feature_type mm --batch_size $mm_batch_size_arg --opt adam --lr_init $lr_init_val --min_lr $lr_min_val --use_min_lr $lr_min_start --epoch $num_epochs --dir $train_dir --aag_audio 0.4 --ckpt $train_dir --steps_per_epoch $mm_steps_per_epoch

echo "Sleeping"
sleep 1
echo "Starting"
train_dir=./train_logs/E2EMLA_emo_sL_r10
python -W ignore main.py --video_frames $video_frames_arg --audio_time $audio_time_arg --feature_type mm --batch_size $mm_batch_size_arg --opt adam --lr_init $lr_init_val --min_lr $lr_min_val --use_min_lr $lr_min_start --epoch $num_epochs --dir $train_dir --aag_audio 0.4 --ckpt $train_dir --steps_per_epoch $mm_steps_per_epoch


echo "Sleeping"
sleep 1
echo "Starting"
train_dir=./train_logs/E2EMLA_emo_sL_r11
python -W ignore main.py --video_frames $video_frames_arg --audio_time $audio_time_arg --feature_type mm --batch_size $mm_batch_size_arg --opt adam --lr_init $lr_init_val --min_lr $lr_min_val --use_min_lr $lr_min_start --epoch $num_epochs --dir $train_dir --aag_audio 0.4 --ckpt $train_dir --steps_per_epoch $mm_steps_per_epoch

echo "Sleeping"
sleep 1
echo "Starting"
train_dir=./train_logs/E2EMLA_emo_sL_r12
python -W ignore main.py --video_frames $video_frames_arg --audio_time $audio_time_arg --feature_type mm --batch_size $mm_batch_size_arg --opt adam --lr_init $lr_init_val --min_lr $lr_min_val --use_min_lr $lr_min_start --epoch $num_epochs --dir $train_dir --aag_audio 0.4 --ckpt $train_dir --steps_per_epoch $mm_steps_per_epoch


