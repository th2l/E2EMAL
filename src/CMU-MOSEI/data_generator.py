"""
Author: Huynh Van Thong
Department of Artificial Intelligence Convergence
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import librosa
import pandas as pd
import numpy as np
import pathlib
from face_extractor import FaceInfo
import tensorflow_hub as hub
import sys
import tqdm
import time
import argparse
import tensorflow_io as tfio
import utils
import copy

DEFAULT_SR = 16000  # Default sampling rate
NUM_MELS = 40  # Mel filter number
WINDOWS_SIZE = 0.025  # filter window size (25ms)
SPECT_STRIDE = 0.01  # STRIDE (10ms)
MAX_FRAME_LENGTH = 400
GET_FACE_FEAT = False
FACE_SIZE = (48, 48)  # (112, 112)
MAX_TEXT_LEN = 30
MAX_TEXT_FEATURE = 300

mp4_root_path = "/mnt/Data/Dataset/CMU_MOSEI/"

# happy	sad	anger	surprise	disgust	fear
def single_file_process(sample_info, part='train', get_mel=False, get_face_feat=False, get_trill=False):
    file_id = sample_info['file_id']

    # Video processing #'vgg16', 'resnet50',
    # video_time = time.time()
    face_exts = ['senet50'] if get_face_feat else None
    face_info = FaceInfo(feature_extract=face_exts, is_show=False).read_video(
        video_path='{}dataset/{}/{}.mp4'.format(mp4_root_path, part, file_id),
        out_shape=FACE_SIZE, num_skip=5,
        resize_input=0.5)
    # Audio processing
    audio, sr = librosa.load('{}dataset/{}/{}.mp4'.format(mp4_root_path, part, file_id))
    if sr != DEFAULT_SR:
        audio = librosa.resample(audio, sr, DEFAULT_SR)

    if get_trill:
        module = hub.load('https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3')
        emb_dict = module(samples=audio, sample_rate=DEFAULT_SR)
        audio = emb_dict['layer19']
    elif get_mel:
        # Mel-Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=DEFAULT_SR, n_fft=int(DEFAULT_SR * WINDOWS_SIZE),
                                                  hop_length=int(DEFAULT_SR * SPECT_STRIDE), n_mels=NUM_MELS)
        # Convert to db
        mel_spec = librosa.power_to_db(mel_spec)
        out_mel = np.zeros((NUM_MELS, MAX_FRAME_LENGTH), dtype=float)
        cur_frame_length = mel_spec.shape[1]
        if cur_frame_length <= MAX_FRAME_LENGTH:
            out_mel[:, :cur_frame_length] = mel_spec[:, :cur_frame_length]
        else:
            start = (cur_frame_length - MAX_FRAME_LENGTH) // 2
            out_mel[:, :MAX_FRAME_LENGTH] = mel_spec[:, start: start + MAX_FRAME_LENGTH]

        audio = out_mel

    # Read word embedding vector of text
    npz = np.load('{}dataset/{}/{}.npy'.format(mp4_root_path, part, file_id))  # Contain word embedding
    glove_embedding_vector = npz  # 300 x
    # if len(glove_embedding_vector) != 300:
    #     print('Check: ', file_id)

    sample_info.update({'video': face_info, 'audio': audio, 'text': glove_embedding_vector})

    return sample_info


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def to_tfrecords(sample_info, part):
    """

    :param sample_info:
    :param part:
    :return:
    """
    # print('Eager mode: ', tf.compat.v1.executing_eagerly())
    data_folder = pathlib.Path('{}dataset/tfrecord/{}'.format(mp4_root_path, part))
    data_folder.mkdir(parents=True, exist_ok=True)
    info_dict = dict()
    for ky in sample_info.keys():
        if isinstance(sample_info[ky], dict):
            for kyky in sample_info[ky].keys():
                serialized_np = tf.io.serialize_tensor(sample_info[ky][kyky])
                info_dict[kyky] = _bytes_feature(serialized_np)
                # info_dict[kyky] = sample_info[ky][kyky]
        else:
            serialized_np = tf.io.serialize_tensor(sample_info[ky])
            info_dict[ky] = _bytes_feature(serialized_np)
            # info_dict[ky] = sample_info[ky]

    # print('Info dict keys: ', info_dict.keys())
    info_dict_example = tf.train.Example(features=tf.train.Features(feature=info_dict))
    with tf.io.TFRecordWriter('{}/{}.tfrecord'.format(data_folder.__str__(), sample_info['file_id'])) as writer:
        writer.write(info_dict_example.SerializeToString())

    # str_features = tf.io.FixedLenFeature([], tf.string)
    # parse_dict = {'file_id': str_features, 's7': str_features, 's2o': str_features,
    #               's2n': str_features, 'bi_emotion': str_features,
    #               'video': str_features, 'audio': str_features, 'text': str_features}
    #
    # for bt in tf.data.TFRecordDataset(['{}/{}.tfrecord'.format(data_folder.__str__(), sample_info['file_id'])]):
    #     example_mess = tf.io.parse_single_example(bt, parse_dict)
    #     pass


def preprocess_audio_signal(raw_audio, remove_noise=True, audio_length=4):
    """

    :param raw_audio:
    :return:
    """
    if remove_noise:
        # Trim the noise
        position = tfio.experimental.audio.trim(raw_audio, axis=0, epsilon=0.1)
        processed = raw_audio[position[0]: position[1]]
    else:
        processed = raw_audio

    # Convert to spectrogram
    spectrogram = tfio.experimental.audio.spectrogram(processed, nfft=512, window=int(WINDOWS_SIZE * DEFAULT_SR),
                                                      stride=int(SPECT_STRIDE * DEFAULT_SR))
    mel_spectrogram = tfio.experimental.audio.melscale(spectrogram, rate=DEFAULT_SR, mels=NUM_MELS, fmin=0,
                                                       fmax=DEFAULT_SR // 2)
    db_mel_spectrogram = tfio.experimental.audio.dbscale(mel_spectrogram, top_db=80)

    out_mel_length = tf.cast(tf.size(db_mel_spectrogram) // NUM_MELS, tf.int32)
    target_mel_length = audio_length * 100
    # num_pad = out_mel_length - target_mel_length
    # st_crop = tf.where(num_pad <= 0, 0, tf.cast(num_pad / 2, tf.int32))
    # ed_crop = tf.where(num_pad <= 0, out_mel_length, st_crop + target_mel_length)
    # ret = tf.pad(db_mel_spectrogram[st_crop: ed_crop, :], tf.constant([[0, target_mel_length], [0, 0]]))[
    #           :target_mel_length, :]

    # Reduce mean to the target shape, padding with zeros if need
    num_padded = out_mel_length + (target_mel_length - out_mel_length % target_mel_length)
    ret = tf.pad(db_mel_spectrogram, tf.constant([[0, target_mel_length], [0, 0]]))[:num_padded, :]
    ret = tf.reshape(ret, (-1, target_mel_length, NUM_MELS))
    ret = tf.reduce_mean(ret, axis=0)

    return ret

def preprocess_text_data(text_vector):
    """

    :param text_vector:
    :param temporal_length:
    :return:
    """
    out_text_length = tf.cast(tf.size(text_vector) // MAX_TEXT_FEATURE, tf.int32)
    num_padded = out_text_length + (MAX_TEXT_LEN - out_text_length % MAX_TEXT_LEN)
    ret = tf.pad(text_vector, tf.constant([[0, MAX_TEXT_LEN], [0, 0]]))[:num_padded, :]
    ret = tf.reshape(ret, (-1, MAX_TEXT_LEN, MAX_TEXT_FEATURE))
    ret = tf.reduce_mean(ret, axis=0)

    return ret

def _parse_function(element, out_dict=None, audio_length=4, video_frames=24, add_lb_in=False):
    """

    :param element:
    :param out_dict:
    :param audio_length: length of audio in second
    :param video_frames: number of frames to be used
    :return:
    """
    str_features = tf.io.FixedLenFeature([], tf.string)
    parse_dict = {'file_id': str_features, 's7': str_features, 's2o': str_features,
                  's2n': str_features, 'bi_emotion': str_features,
                  'video': str_features, 'audio': str_features, 'text': str_features}

    example_mess = tf.io.parse_single_example(element, parse_dict)

    if out_dict is None:
        audio = tf.io.parse_tensor(example_mess['audio'], tf.float32)[:10]
        return audio
    out_parser = dict()
    out_label_parser = dict()

    list_labels = []
    for ky in out_dict:
        if ky in ['audio', 'text', 'int']:
            list_labels.append(ky)
        elif ky == 'video':
            list_labels.append('face')

    for ky in parse_dict.keys():
        if ky in out_dict:
            if ky == 'file_id':
                out_parser[ky] = tf.io.parse_tensor(example_mess[ky], tf.string)
                # tf.print(out_parser[ky])
            if ky == 's7':
                # Sentiment
                curs7 = tf.one_hot(tf.io.parse_tensor(example_mess[ky], tf.int32), depth=7)
                curs7.set_shape((7, ))
                for dt_spec in list_labels:
                    out_label_parser['{}_{}'.format(dt_spec, ky)] = curs7
                    if add_lb_in:
                        out_parser['{}_{}_lb'.format(dt_spec, ky)] = curs7

            elif ky == 's2o' or ky == 's2n':
                # Sentiment
                curs2 = tf.one_hot(tf.io.parse_tensor(example_mess[ky], tf.int32), depth=2)
                curs2.set_shape((2,))
                for dt_spec in list_labels:
                    out_label_parser['{}_{}'.format(dt_spec, ky)] = curs2
                    if add_lb_in:
                        out_parser['{}_{}_lb'.format(dt_spec, ky)] = curs2

            elif ky == 'bi_emotion':
                cur_emo = tf.io.parse_tensor(example_mess[ky], tf.float32)
                cur_emo.set_shape((6, ))
                for dt_spec in list_labels:
                    out_label_parser['{}_{}'.format(dt_spec, ky)] = cur_emo
                    if add_lb_in:
                        out_parser['{}_{}_lb'.format(dt_spec, ky)] = cur_emo

                # out_label_parser[ky] = cur_emo
                # out_label_parser[ky].set_shape((6,))
                #
                # if add_lb_in:
                #     out_parser['{}_lb'.format(ky)] = out_label_parser[ky]

            elif ky == 'video':
                read_video = tf.io.parse_tensor(example_mess[ky], tf.uint8)

                # tf.pad(text_vector, tf.constant([[0, MAX_TEXT_LEN], [0, 0]]))[:num_padded, :]
                st_crop_vd = tf.cast((tf.size(read_video) / (FACE_SIZE[0] * FACE_SIZE[1] * 3) - video_frames) / 2,
                                     tf.int32)

                num_frames_inp = tf.cast(tf.size(read_video) / (FACE_SIZE[0] * FACE_SIZE[1] * 3), tf.int32)
                st_crop = tf.where(tf.less(num_frames_inp, video_frames), 0, st_crop_vd)

                read_video_pad = tf.pad(read_video, tf.constant([[0, video_frames], [0, 0], [0, 0], [0, 0]]))
                out_parser[ky] = tf.cast(read_video_pad[st_crop: st_crop + video_frames, :, :, :], tf.float32)

                out_parser[ky].set_shape((video_frames, FACE_SIZE[0], FACE_SIZE[1], 3))

            elif ky == 'audio':
                read_audio = tf.io.parse_tensor(example_mess[ky], tf.float32)
                out_parser[ky] = preprocess_audio_signal(read_audio, remove_noise=False, audio_length=audio_length)

            elif ky == 'text':
                read_text = tf.io.parse_tensor(example_mess[ky], tf.double)
                out_parser[ky] = preprocess_text_data(read_text)
                # out_parser[ky] = tf.pad(read_text, tf.constant([[0, MAX_TEXT_LEN], [0, 0]]))[:MAX_TEXT_LEN, :]

    if add_lb_in:
        list_keys_label = list(out_label_parser.keys())
        for ky in list_keys_label:
            if not ky.endswith('lb') and 'int' not in ky:
                out_label_parser.pop(ky, None)
    else:
        list_keys_label = list(out_label_parser.keys())
        for ky in list_keys_label:
            if 'int' not in ky:
                out_label_parser.pop(ky, None)

    return out_parser, out_label_parser

def audio_frequency_masking(features, labels, p=0.5):
    """
    Audio frequency_masking augmentation
    :param element: tuple of (features, labels)
    :return:
    """

    def freq_mask(input, param, name=None):
        """
        Apply masking to a spectrogram in the freq domain.

        Args:
          input: An audio spectogram.
          param: Parameter of freq masking.
          name: A name for the operation (optional).
        Returns:
          A tensor of spectrogram.
        """
        # TODO: Support audio with channel > 1.
        freq_max = tf.shape(input)[1]
        f = tf.random.uniform(shape=(), minval=0, maxval=param, dtype=tf.dtypes.int32)
        f0 = tf.random.uniform(
            shape=(), minval=0, maxval=freq_max - f, dtype=tf.dtypes.int32
        )
        indices = tf.reshape(tf.range(freq_max), (1, -1))
        condition = tf.math.logical_and(
            tf.math.greater_equal(indices, f0), tf.math.less(indices, f0 + f)
        )
        return tf.where(condition, 0., input)

    if 'audio' in features.keys() and p > 0.:
        if tf.random.uniform([]) < p:
            features['audio'] = freq_mask(features['audio'], param=5)

    return features, labels


def audio_time_masking(features, labels, p=0.5):
    """
    Audio time masking augmentation
    :param element:
    :param p:
    :return:
    """

    def time_mask(input, param, name=None):
        """
        Apply masking to a spectrogram in the time domain.

        Args:
          input: An audio spectogram.
          param: Parameter of time masking.
          name: A name for the operation (optional).
        Returns:
          A tensor of spectrogram.
        """
        # TODO: Support audio with channel > 1.
        time_max = tf.shape(input)[0]
        t = tf.random.uniform(shape=(), minval=0, maxval=param, dtype=tf.dtypes.int32)
        t0 = tf.random.uniform(
            shape=(), minval=0, maxval=time_max - t, dtype=tf.dtypes.int32
        )
        indices = tf.reshape(tf.range(time_max), (-1, 1))
        condition = tf.math.logical_and(
            tf.math.greater_equal(indices, t0), tf.math.less(indices, t0 + t)
        )
        return tf.where(condition, 0., input)

    if 'audio' in features.keys() and p > 0.:
        if tf.random.uniform([]) < p:
            features['audio'] = time_mask(features['audio'], param=10)

    return features, labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data generator')
    parser.add_argument('--start_from', type=float, default=0, required=False, help='Start from')
    args = parser.parse_args()
    print('Start from: ', args.start_from)

    parts = ['train', 'val', 'test']
    utils.set_gpu_growth_or_cpu(use_cpu=False)

    # str_features = tf.io.FixedLenFeature([], tf.string)
    # parse_dict = {'file_id': str_features, 's7': str_features, 's2o': str_features,
    #               's2n': str_features, 'bi_emotion': str_features,
    #               'video': str_features, 'audio': str_features, 'text': str_features}
    #
    # out_dict = ['video', 'audio', 'text', 'bi_emo']
    # for bt in tf.data.TFRecordDataset(['{}dataset/tfrecord/{}/{}.tfrecord'.format(mp4_root_path, 'train', '-3g5yACwYnA_0')]):
    #     # example_mess = tf.io.parse_single_example(bt, parse_dict)
    #     ret = _parse_function(bt, out_dict)
    #     pass
    #
    # sys.exit(0)
    for part in parts:
        print('Part: ', part)
        list_emo = []
        csv_integrated = pd.read_csv('{}dataset/{}.csv'.format(mp4_root_path, part)).values

        num_samples = csv_integrated.shape[0]

        for idx in tqdm.tqdm(range(int(num_samples * args.start_from), num_samples)):
            mp4_name = csv_integrated[idx, 0]

            if os.path.isfile('{}dataset/tfrecord/{}/{}.tfrecord'.format(mp4_root_path, part, mp4_name)):
                continue
                # check_tf = tf.data.TFRecordDataset(
                #     '{}dataset/tfrecord/{}/{}.tfrecord'.format(mp4_root_path, part, mp4_name)).map(_parse_function)
                # ok = True
                # try:
                #     for dm in check_tf:
                #         ok = True
                # except Exception as e:
                #     ok = False
                #     print('{}dataset/tfrecord/{}/{}.tfrecord'.format(mp4_root_path, part, mp4_name))
                # if ok:
                #     continue
            s7 = csv_integrated[idx, 1]
            s2o = csv_integrated[idx, 2]
            s2n = csv_integrated[idx, 3]
            bi_emotion = csv_integrated[idx, 4:]

            sample_info = {'file_id': mp4_name, 's7': s7, 's2o': s2o, 's2n': s2n, 'bi_emotion': bi_emotion.astype(np.float32)}
            sample_info_feat = single_file_process(sample_info, part, get_face_feat=GET_FACE_FEAT, get_trill=False)
            to_tfrecords(sample_info_feat, part)

    pass
