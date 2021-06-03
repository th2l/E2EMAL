import gc
import os
import sys
from copy import deepcopy
from shutil import copyfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import pandas as pd
import pathlib
import argparse
import utils
from functools import partial
from data_generator import _parse_function, audio_frequency_masking, audio_time_masking
from tqdm import tqdm
import random, math
import models
import numpy as np
import tensorflow_addons as tfa
import configparser
from sklearn.model_selection import StratifiedKFold, KFold
import joblib

emo_dict = ['hap', 'sad', 'ang', 'sur', 'dis', 'fea', 'neu']


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
        out_dict.append('speech_emo')
    if 'video' in args.feature_type:
        out_dict.append('video')
        out_dict.append('face_emo')
    if 'text' in args.feature_type:
        out_dict.append('text')
        if 'speech_emo' not in out_dict:
            out_dict.append('text_emo')

    if 'mm' in args.feature_type:
        if config['multimodal'].getboolean('multi_outputs'):
            out_dict = out_dict + ['audio', 'video', 'text', 'int_emo', 'speech_emo', 'face_emo', 'text_emo']
        else:
            out_dict = out_dict + ['audio', 'video', 'text', 'int_emo']

    return out_dict


def get_by_ids(use_ids, list_ids_tfrecords):
    """

    :param use_data: list of path to tfrecords
    :param use_ids: list of ids
    :param list_ids_tfrecords: videos corresponding to each ids
    :return:
    """

    ret_data = []
    for ids in use_ids:
        ret_data = ret_data + list_ids_tfrecords[ids].tolist()

    return sorted(ret_data)


def get_cv_data(n_splits=5, seed=12, byParticipantID=False):
    """
    Merge train and validation, and random split for cross validation
    :return:
    """
    if not byParticipantID:
        emo_dict = ['hap', 'sad', 'ang', 'sur', 'dis', 'fea', 'neu']
        train_csv = pd.read_csv('2020-1/train.csv').values
        val_csv = pd.read_csv('2020-1/val.csv').values

        train_data = [('2020-1/tfrecord/train/{}.tfrecord'.format(train_csv[idx, 0]), emo_dict.index(train_csv[idx, 1]))
                      for
                      idx in range(train_csv.shape[0])]
        val_data = [('2020-1/tfrecord/val/{}.tfrecord'.format(val_csv[idx, 0]), emo_dict.index(val_csv[idx, 1])) for
                    idx in range(val_csv.shape[0])]
        merged_data = np.concatenate([train_data, val_data])

        merged_id = merged_data[:, 0].reshape(-1, 1)
        merged_label = merged_data[:, 1].astype(np.int8)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_indexing = []
        for train_index, val_index in skf.split(merged_id, merged_label):
            fold_indexing.append((merged_id[train_index, 0].flatten(), merged_id[val_index, 0].flatten()))

    else:
        # Split by participant ID
        participant_ids = np.load('train_val_ids_dict.npy', allow_pickle=True).item()
        list_ids = np.array(sorted(list(participant_ids.keys())))
        skf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_indexing = []
        for train_index, val_index in skf.split(list_ids):
            train_ids = get_by_ids(list_ids[train_index], participant_ids)
            val_ids = get_by_ids(list_ids[val_index], participant_ids)
            fold_indexing.append((train_ids, val_ids))

    return fold_indexing


def load_data(args, train_id=None, val_id=None, load_test=True, load_train=True):
    """

    :param args:
    :param train_id:
    :param val_id:
    :param load_test:
    :return:
    """

    autotune = tf.data.experimental.AUTOTUNE
    batch_size = args.batch_size

    out_dict = get_out_dict(args)

    add_lb_in = ('mm' in args.feature_type and is_multi_outputs())

    parse_func = partial(_parse_function, out_dict=out_dict, audio_length=args.audio_time,
                         video_frames=args.video_frames, add_lb_in=add_lb_in)

    audio_time_masking_func = partial(audio_time_masking, p=args.aag_audio)
    audio_frequency_masking_func = partial(audio_frequency_masking, p=args.aag_audio)

    if load_train:
        if train_id is None:
            train_list = sorted(
                ['2020-1/tfrecord/train/{}.tfrecord'.format(int_id) for int_id in
                 pd.read_csv('2020-1/train.csv').values[:, 0]])
        else:
            train_list = sorted(train_id)

        random.shuffle(train_list)

        if val_id is None:
            val_list = sorted(
                ['2020-1/tfrecord/val/{}.tfrecord'.format(int_id) for int_id in pd.read_csv('2020-1/val.csv').values[:, 0]])
        else:
            val_list = sorted(val_id)


        loaders = {
            'train': tf.data.TFRecordDataset(train_list).shuffle(args.buffer_size, reshuffle_each_iteration=True,
                                                                 seed=args.seed).repeat().map(parse_func,
                                                                                              num_parallel_calls=autotune).map(
                audio_frequency_masking_func, num_parallel_calls=autotune).map(audio_time_masking_func,
                                                                               num_parallel_calls=autotune).batch(
                batch_size).prefetch(autotune),
            'val': tf.data.TFRecordDataset(val_list).map(parse_func, num_parallel_calls=autotune).batch(
                batch_size).prefetch(autotune),
        }
        loaders_len = {'train': len(train_list), 'val': len(val_list)}
    else:
        loaders = {}
        loaders_len = {}

    if load_test:
        test1_list = sorted([x.__str__() for x in pathlib.Path('2020-1/tfrecord/test1/').glob('*.tfrecord')])
        test2_list = sorted([x.__str__() for x in pathlib.Path('2020-1/tfrecord/test2/').glob('*.tfrecord')])
        test3_list = sorted([x.__str__() for x in pathlib.Path('2020-1/tfrecord/test3/').glob('*.tfrecord')])
        test_out_dict = get_out_dict(args, is_test=True)
        test_parse_func = partial(_parse_function, out_dict=test_out_dict, audio_length=args.audio_time,
                                  video_frames=args.video_frames, add_lb_in=add_lb_in)

        loaders.update({
            'test1': tf.data.TFRecordDataset(test1_list).map(test_parse_func, num_parallel_calls=autotune).batch(
                batch_size).prefetch(autotune),
            'test2': tf.data.TFRecordDataset(test2_list).map(test_parse_func, num_parallel_calls=autotune).batch(
                batch_size).prefetch(autotune),
            'test3': tf.data.TFRecordDataset(test3_list).map(test_parse_func, num_parallel_calls=autotune).batch(
                batch_size).prefetch(autotune)
        })

        loaders_len.update({'test1': len(test1_list), 'test2': len(test2_list), 'test3': len(test3_list)})

    return loaders, loaders_len


def get_base_model(args):
    """
    Create model
    :param args:
    :return:
    """
    if 'audio' in args.feature_type:
        base_model = models.audio_model(args)
    elif 'video' in args.feature_type:
        base_model = models.video_model(args)
    elif 'text' in args.feature_type:
        base_model = models.text_model(args)
    else:
        base_model = models.multimodal_model(args)

    return base_model


def is_multi_outputs():
    config = configparser.ConfigParser()
    config.read('models_config.cfg')

    return config['multimodal'].getboolean('multi_outputs')


def train(args, base_model=None, loaders=None, loaders_len=None, is_fold=-1, is_train=True):
    """

    :param args:
    :return:
    """
    if loaders is None:
        loaders, loaders_len = load_data(args)

    print(loaders_len)

    postfix = '' if is_fold == -1 else '-kf{}'.format(is_fold)
    # if args.batch_size == 1:
    #     for dm in loaders['train']:
    #         print(dm[0].keys(), dm[1].keys())
    #         return
    #         continue
    #     return
    if base_model is None:
        base_model = get_base_model(args)

    train_steps_per_epoch = int(args.steps_per_epoch * (loaders_len['train'] // args.batch_size))
    val_steps_per_epoch = math.ceil(loaders_len['val'] / args.batch_size)

    lr_schedule = utils.CusLRScheduler(initial_learning_rate=args.lr_init, min_lr=args.min_lr, lr_start_warmup=0.,
                                       warmup_steps=5 * train_steps_per_epoch, num_constant=0 * train_steps_per_epoch,
                                       T_max=args.use_min_lr * train_steps_per_epoch, num_half_cycle=1)

    if args.opt == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

    opt_swa = tfa.optimizers.SWA(opt, start_averaging=int(args.use_min_lr * train_steps_per_epoch),
                                 average_period=int(args.use_min_lr * 1.))

    loss_obj = None if 'mm' in args.feature_type else tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.3,
                                                                                              from_logits=True)
    metric_obj = tf.keras.metrics.CategoricalAccuracy(name='acc')

    base_model.compile(optimizer=opt_swa, loss=[loss_obj], metrics=[metric_obj])

    monitor_name = 'val_acc'  #
    pathlib.Path(args.dir).mkdir(parents=True, exist_ok=True)
    best_ckpt = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(args.dir, '{}_best_checkpoint{}.h5'.format(args.feature_type[0], postfix)),
        monitor=monitor_name,
        verbose=0, save_best_only=True, save_weights_only=True,
        mode='max')
    tsb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args.dir, 'tsb'))
    cbacks = [tsb_callback]  # [best_ckpt, ]

    if is_train:
        base_model.fit(loaders['train'], validation_data=loaders['val'], epochs=args.epoch,
                       steps_per_epoch=train_steps_per_epoch,
                       validation_steps=val_steps_per_epoch,
                       callbacks=cbacks, verbose=1)

        opt_swa.assign_average_vars(base_model.variables)
        base_model.save_weights(os.path.join(args.dir, '{}_best_checkpoint{}.h5'.format(args.feature_type[0], postfix)))
    else:
        postfix = postfix + '_testOnly'

    base_model.load_weights(os.path.join(args.dir, '{}_best_checkpoint{}.h5'.format(args.feature_type[0], postfix)))
    res = base_model.evaluate(loaders['val'])
    print("Evaluating ", res)

    return base_model


def run_cv_train(args, n_splits=5):
    """

    :param args:
    :param n_splits:
    :return:
    """
    load_test = True
    data_loaders = {}
    data_loaders_len = {}
    base_model = None
    byParticipantID = False
    for idx in range(n_splits):
        cv_data_indexing = get_cv_data(n_splits=n_splits, seed=args.seed + 11 * idx + 1,
                                       byParticipantID=byParticipantID)

        cur_train_id, cur_val_id = cv_data_indexing[idx]
        cur_loaders, cur_loaders_len = load_data(args, train_id=cur_train_id, val_id=cur_val_id, load_test=load_test)
        load_test = False

        data_loaders.update(cur_loaders)
        data_loaders_len.update(cur_loaders_len)

        base_model = train(args, base_model=base_model, loaders=data_loaders, loaders_len=data_loaders_len, is_fold=idx,
                           is_train=True)

        # Reset session
        base_model = None

        data_loaders['train'] = None
        data_loaders['val'] = None
        gc.collect()

        utils.set_seed(args.seed, reset_session=True)

    test(args, is_fold=True)


def generate_predictions_kfold(list_models, data_loader, is_test='val', write_name='data_write.txt'):
    """

    :param list_models:
    :param data_loader:
    :param is_test:
    :param write_name:
    :return:
    """
    if len(list_models) < 1:
        raise ValueError('Need to Specify models')

    out_res = []
    for data in data_loader:
        data_pred_list = []
        for idx in range(len(list_models)):
            cur_preds = list_models[idx].predict(data[0])  # ['int_emo']
            if isinstance(cur_preds, dict):
                data_pred_list.append(cur_preds['int_emo'])
            else:
                data_pred_list.append(cur_preds)
        data_pred = np.mean(np.stack(data_pred_list), axis=0)
        data_pred = np.argmax(data_pred, axis=-1)

        for idx in range(len(list_models)):
            data_pred_list[idx] = np.argmax(data_pred_list[idx], axis=-1)

        data_id = data[0]['file_id'].numpy()

        cur_res = []
        for idx in range(data_id.shape[0]):
            cur_tuple = [data_id[idx], emo_dict[data_pred[idx]]]
            for idx_m in range(len(list_models)):
                cur_tuple.append(emo_dict[data_pred_list[idx_m][idx]])

            cur_res.append(cur_tuple)

        out_res = out_res + cur_res

    out_res = np.array(out_res)
    cols = ['FileID', 'Emotion']
    for idx in range(len(list_models)):
        cols.append('Emotion{}'.format(idx))
    pd.DataFrame(out_res,
                 columns=cols).to_csv(
        write_name[:-4] + '_all.csv', sep=',', index=False)
    pd.DataFrame(out_res[:, [0, 1]],
                 columns=['FileID', 'Emotion']).to_csv(write_name, sep=',', index=False)


def generate_predictions(model, data_loader, is_test='val', write_name='data_write.txt'):
    """
    Generate prediction
    :param model:
    :param loader:
    :return:
    """

    emo_dict = ['hap', 'sad', 'ang', 'sur', 'dis', 'fea', 'neu']
    out_res = []
    for data in data_loader:
        data_pred = model.predict(data[0])  # ['int_emo']
        data_pred = np.argmax(data_pred, axis=-1)

        if 'test' in is_test:
            data_id = data[0]['file_id'].numpy()
            cur_res = [(data_id[idx], emo_dict[data_pred[idx]]) for idx in range(data_id.shape[0])]
        else:
            data_emo = np.argmax(data[1]['int_emo'].numpy(), axis=-1)
            # cur_res = [(emo_dict.index(data_emo[idx]), data_pred[idx]) for idx in range(data_emo.shape[0])]
            cur_res = [(data_emo[idx], data_pred[idx]) for idx in range(data_emo.shape[0])]

        out_res = out_res + cur_res

    out_res = np.array(out_res)
    pd.DataFrame(out_res, columns=['FileID', 'Emotion']).to_csv(write_name, sep=',', index=False)


def write_predictions(base_model, data_loaders, postfix=''):
    """

    :param model:
    :param data_loaders:
    :return:
    """
    # print('Generate val predictions')
    # generate_predictions(base_model, data_loaders['val'], is_test='val',
    #                      write_name='{}/val{}.csv'.format(args.dir, postfix))
    print('Generate test1 predictions')
    generate_predictions(base_model, data_loaders['test1'], is_test='test1',
                         write_name='{}/test1{}.csv'.format(args.dir, postfix))

    if 'test2' in data_loaders.keys():
        print('Generate test2 predictions')
        generate_predictions(base_model, data_loaders['test2'], is_test='test2',
                             write_name='{}/test2{}.csv'.format(args.dir, postfix))
    if 'test3' in data_loaders.keys():
        print('Generate test3 predictions')
        generate_predictions(base_model, data_loaders['test3'], is_test='test2',
                             write_name='{}/test3{}.csv'.format(args.dir, postfix))


def test(args, is_fold=True):
    """

    :param args:
    :return:
    """
    loaders, _ = load_data(args, load_train=False)
    num_models = 5
    if not is_fold:
        base_model = get_base_model(args)
        base_model.compile(metrics=tf.keras.metrics.CategoricalAccuracy(name='acc'))
        base_model.load_weights('{}/{}_best_checkpoint.h5'.format(args.dir, 'mm'))

        val_res = base_model.evaluate(loaders['val'])
        print('Evaluate val res: ', val_res)

        write_predictions(base_model, loaders)

        print('Completed generate text prediction')
    else:
        list_models = []
        for kf in range(num_models):
            cur_model = get_base_model(args)
            cur_model.load_weights(
                os.path.join(args.dir, '{}_best_checkpoint-kf{}.h5'.format(args.feature_type[0], kf)))
            list_models.append(cur_model)

        postfix = 'kf_fuse_mean'
        generate_predictions_kfold(list_models, loaders['test1'], is_test='test1',
                                   write_name='{}/test1-{}.csv'.format(args.dir, postfix))
        generate_predictions_kfold(list_models, loaders['test2'], is_test='test2',
                                   write_name='{}/test2-{}.csv'.format(args.dir, postfix))
        if 'test3' in loaders.keys():
            generate_predictions_kfold(list_models, loaders['test3'], is_test='test3',
                                       write_name='{}/test3-{}.csv'.format(args.dir, postfix))


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
    tf.config.optimizer.set_jit(True)
    # Mixed precision
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_policy(policy)
    # tf.logging.set_verbosity(tf.logging.ERROR)

    # tf.compat.v1.disable_eager_execution()
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
        copyfile('./data_generator.py',
                 os.path.join(args.dir, "src", "data_generator_{}.py".format(args.feature_type[0])))
        copyfile('./face_extractor.py',
                 os.path.join(args.dir, "src", "face_extractor_{}.py".format(args.feature_type[0])))
        copyfile('./train.sh', os.path.join(args.dir, "src", "train_{}.sh".format(args.feature_type[0])))

    except:
        print("Can not copy files")

    with open('{}/{}_arg.txt'.format(args.dir, args.feature_type[0]), 'w') as fd:
        fd.write(str(vars(args)))

    if args.test != '':
        print('Testing progress')
        test(args)
    else:
        # tf.compat.v1.disable_eager_execution()
        print("TF version ", tf.__version__)
        # train(args)
        run_cv_train(args)
        print('Training progress')
