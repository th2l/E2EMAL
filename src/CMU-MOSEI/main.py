import os
from contextlib import redirect_stdout

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import configparser
import math

import pathlib
import sys
from shutil import copyfile

import tensorflow as tf
from functools import partial

import models
from data_generator import _parse_function, audio_frequency_masking, audio_time_masking, mp4_root_path
import pandas as pd
import utils

category_dict = {'s7': 's7', 's2o': 's2o', 's2n': 's2n', 'bi_emotion': 'bi_emotion'}


def is_multi_outputs():
    config = configparser.ConfigParser()
    config.read('models_config.cfg')

    return config['multimodal'].getboolean('multi_outputs')


def get_out_dict(args, is_test=False):
    """

    :param args:
    :return:
    """
    out_dict = [] if not is_test else ['file_id']
    config = configparser.ConfigParser()
    config.read('models_config.cfg')

    if 'audio' in args.feature_type:
        out_dict.append('audio')
    if 'video' in args.feature_type:
        out_dict.append('video')
    if 'text' in args.feature_type:
        out_dict.append('text')

    category_key = config['category'].get('category')
    out_dict.append(category_dict[category_key])

    print("Problem category: ", category_dict[category_key])
    if 'mm' in args.feature_type:
        if config['multimodal'].getboolean('multi_outputs'):
            out_dict = out_dict + ['int', 'audio', 'video', 'text', category_dict[category_key]]
        else:
            out_dict = out_dict + ['int', 'audio', 'video', 'text', category_dict[category_key]]

    return out_dict


def load_data(args, load_test=True, load_train=True, preset_id=None):
    """

    :param args:
    :param load_test:
    :param load_train:
    :return:
    """
    print("Loading data")
    autotune = tf.data.experimental.AUTOTUNE
    batch_size = args.batch_size
    add_lb_in = ('mm' in args.feature_type and is_multi_outputs())
    out_dict = get_out_dict(args)

    if 's7' in out_dict:
        num_classes = 7
    elif 's2o' in out_dict or 's2n' in out_dict:
        num_classes = 2
    else:
        num_classes = 6

    parse_func = partial(_parse_function, out_dict=out_dict, audio_length=args.audio_time,
                         video_frames=args.video_frames, add_lb_in=add_lb_in)
    audio_time_masking_func = partial(audio_time_masking, p=args.aag_audio)
    audio_frequency_masking_func = partial(audio_frequency_masking, p=args.aag_audio)

    loaders = {}
    loaders_len = {}
    for split in ['train', 'val', 'test']:
        if preset_id is None or split not in preset_id:
            data_csv = pd.read_csv('{}dataset/{}.csv'.format(mp4_root_path, split))['Name'].values
        else:
            data_csv = preset_id[split]

        data_list = ['{}dataset/tfrecord/{}/{}.tfrecord'.format(mp4_root_path, split, vid) for vid in data_csv]

        if split == 'train':
            loaders[split] = tf.data.TFRecordDataset(data_list).shuffle(args.buffer_size, reshuffle_each_iteration=True,
                                                                        seed=args.seed).repeat().map(parse_func,
                                                                                                     num_parallel_calls=autotune).map(
                audio_frequency_masking_func, num_parallel_calls=autotune).map(audio_time_masking_func,
                                                                               num_parallel_calls=autotune).batch(
                batch_size).prefetch(autotune)
        else:
            loaders[split] = tf.data.TFRecordDataset(data_list).map(parse_func, num_parallel_calls=autotune).batch(
                batch_size).prefetch(autotune)

        loaders_len[split] = len(data_list)
    #
    # for dm in loaders['train']:
    #     print(type(dm))
    #     break
    return loaders, loaders_len, num_classes


def get_kfold_id():
    train_csv = pd.read_csv('{}dataset/{}.csv'.format(mp4_root_path, 'train'))['Name']
    val_csv = pd.read_csv('{}dataset/{}.csv'.format(mp4_root_path, 'val'))['Name']
    pass


def get_base_model(args, num_classes):
    """
    Create model
    :param args:
    :return:
    """

    if 'audio' in args.feature_type:
        base_model = models.audio_model(args, num_classes=num_classes)
    elif 'video' in args.feature_type:
        base_model = models.video_model(args, num_classes=num_classes)
    elif 'text' in args.feature_type:
        base_model = models.text_model(args, num_classes=num_classes)
    else:
        base_model = models.multimodal_model(args, num_classes=num_classes)

    return base_model


def train(args, base_model=None, loaders=None, loaders_len=None, is_train=True):
    """

    :param args:
    :param base_model:
    :param loaders:
    :param loaders_len:
    :param is_train:
    :return:
    """

    if loaders is None:
        loaders, loaders_len, num_classes = load_data(args)
    else:
        num_classes = 7
    print(loaders_len)

    if base_model is None:
        base_model = get_base_model(args, num_classes=num_classes)

    print("Training progress")
    postfix = ''
    with open(os.path.join(args.dir, 'src', 'model_summary{}.txt'.format(postfix)), 'w') as f:
        with redirect_stdout(f):
            base_model.summary()

    base_model.summary()

    train_steps_per_epoch = int(args.steps_per_epoch * (loaders_len['train'] // args.batch_size))
    val_steps_per_epoch = math.ceil(loaders_len['val'] / args.batch_size)

    lr_schedule = utils.CusLRScheduler(initial_learning_rate=args.lr_init, min_lr=args.min_lr, lr_start_warmup=0.,
                                       warmup_steps=5 * train_steps_per_epoch, num_constant=0 * train_steps_per_epoch,
                                       T_max=args.use_min_lr * train_steps_per_epoch, num_half_cycle=1)

    if args.opt == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

    if num_classes == 6:
        """ Binary emotion - multilabel"""
        loss_func = utils.MoseiEmotionLoss()
        metric_obj = {'acc': utils.WeightedAccuracy(num_classes=6, threshold=0.5),
                      'f1score': utils.F1ScoreMetric(num_classes=6, average='weighted', threshold=0.5)}
    else:
        loss_func = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0, from_logits=True)
        metric_obj = tf.keras.metrics.CategoricalAccuracy(name='acc')

    loss_obj = None if 'mm' in args.feature_type and is_multi_outputs() else loss_func

    base_model.compile(optimizer=opt, loss=[loss_obj], metrics=[metric_obj])
    if num_classes == 6:
        monitor_name = 'val_wAcc'
    else:
        monitor_name = 'val_acc'
    # monitor_name = 'val_loss'
    pathlib.Path(args.dir).mkdir(parents=True, exist_ok=True)
    best_ckpt = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(args.dir, '{}_best_checkpoint{}.h5'.format(args.feature_type[0], postfix)),
        monitor=monitor_name,
        verbose=0, save_best_only=True, save_weights_only=True,
        mode='max')
    # mode='min')

    tsb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args.dir, 'tsb'))
    # cbacks = [tsb_callback]  # [best_ckpt, ]
    cbacks = [tsb_callback, best_ckpt]

    if is_train:
        val_data = loaders['val']  # None or loaders['val']
        base_model.fit(loaders['train'], validation_data=val_data, epochs=args.epoch,
                       steps_per_epoch=train_steps_per_epoch,
                       validation_steps=val_steps_per_epoch,
                       callbacks=cbacks, verbose=1)

        # base_model.save_weights(os.path.join(args.dir, '{}_best_checkpoint{}.h5'.format(args.feature_type[0], postfix)))
    else:
        postfix = postfix + '_testOnly'

    base_model.load_weights(os.path.join(args.dir, '{}_best_checkpoint{}.h5'.format(args.feature_type[0], postfix)))

    val_res = evaluate(base_model, loaders['val'], split='val', write_csv=True)
    test_res = evaluate(base_model, loaders['test'], split='test', write_csv=True)

    with open(os.path.join(args.dir, 'src', 'model_summary{}.txt'.format(postfix)), 'a') as f:
        f.write('Val performance (Loss, OverallWacc, OverallF1, PerClassWAcc, PerClassF1: ' + ' '.join(str(x) for x in val_res) + '\n')
        f.write('Test performance (Loss, OverallWacc, OverallF1, PerClassWAcc, PerClassF1: ' + ' '.join(str(x) for x in test_res) + '\n')

    with open(os.path.join(args.dir, 'results{}.txt'.format(postfix)), 'a') as f:
        f.write('Val performance (Loss, OverallWacc, OverallF1, PerClassWAcc, PerClassF1: ' + ' '.join(
            str(x) for x in val_res) + '\n')
        f.write('Test performance (Loss, OverallWacc, OverallF1, PerClassWAcc, PerClassF1: ' + ' '.join(
            str(x) for x in test_res) + '\n')


def evaluate(base_model, loader, split, write_csv=False):
    """

    :param base_model:
    :param loader:
    :param split:
    :param write_csv:
    :return:
    """
    print('Evaluating {}: '.format(split))
    res = base_model.evaluate(loader)
    per_class = []
    for m in base_model.metrics:
        if m.name in ['wAcc', 'f1score']:
            per_class = per_class + list(m.per_class_result().numpy())
    res = res + per_class
    print('performance (Loss, OverallWacc, OverallF1, PerClassWAcc, PerClassF1: {} '.format(res))

    if write_csv:
        print('Write prediction to csv')
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multimodal Emotion Recognition')
    parser.add_argument('--seed', type=int, default=1, help='seed (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size (default: 32)')
    parser.add_argument('--buffer_size', type=int, default=10240, help='buffer size (default: 4096)')
    parser.add_argument('--audio_time', type=int, default=4, help='time of audio to be use (default: 4 seconds)')
    parser.add_argument('--video_frames', type=int, default=16, help='Number of frames to be used in each video')
    parser.add_argument('--feature_type', nargs='+', help='Feature to be used', required=True)
    parser.add_argument('--steps_per_epoch', type=float, default=1., help='Number of steps per epoch (default: 1.)')
    parser.add_argument('--epoch', type=int, default=10, help='Number of epochs (default: 10)')

    parser.add_argument('--aag_audio', type=float, default=0., help='Audio augmentation probability (default: 0.)')

    parser.add_argument('--lr_init', type=float, default=1e-3, help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Initial learning rate (default: 1e-5)')
    parser.add_argument('--use_min_lr', type=int, default=15, help='Start to use min lr (default: 20)')
    parser.add_argument('--opt', type=str, default='adam', help='Optimizer (default: adam)')

    parser.add_argument('--test', type=str, default='', help='Test (multi-modal) checkpoint (default: "")')
    parser.add_argument('--ckpt', type=str, default='', help='Root model checkpoint (default: "")')

    parser.add_argument('--dir', type=str, default='./tmp', help='Training logs directory (default: tmp)')

    args = parser.parse_args()

    print("Feature: ", args.feature_type[0])
    utils.set_gpu_growth_or_cpu(use_cpu=False, write_info=True)
    utils.set_seed(args.seed)

    pathlib.Path(os.path.join(args.dir, "src")).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.dir, 'command_{}.sh'.format(args.feature_type[0])), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')
        f.write('\n')
    try:
        copyfile('main.py', os.path.join(args.dir, "src", "main_{}.py".format(args.feature_type[0])))
        copyfile('./utils.py', os.path.join(args.dir, "src", "utils_{}.py".format(args.feature_type[0])))
        copyfile('./models.py', os.path.join(args.dir, "src", "models_{}.py".format(args.feature_type[0])))
        copyfile('./models_config.cfg',
                 os.path.join(args.dir, "src", "models_config_{}.cfg".format(args.feature_type[0])))
        copyfile('data_generator.py',
                 os.path.join(args.dir, "src", "data_generator_{}.py".format(args.feature_type[0])))
        copyfile('./face_extractor.py',
                 os.path.join(args.dir, "src", "face_extractor_{}.py".format(args.feature_type[0])))
        copyfile('./train.sh', os.path.join(args.dir, "src", "train_{}.sh".format(args.feature_type[0])))

    except:
        print("Can not copy files")

    with open('{}/{}_arg.txt'.format(args.dir, args.feature_type[0]), 'w') as fd:
        fd.write(str(vars(args)))

    train(args)
