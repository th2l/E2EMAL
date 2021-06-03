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

DEFAULT_SR = 16000  # Default sampling rate
NUM_MELS = 40  # Mel filter number
WINDOWS_SIZE = 0.025  # filter window size (25ms)
SPECT_STRIDE = 0.01  # STRIDE (10ms)
MAX_FRAME_LENGTH = 400
GET_FACE_FEAT = False
FACE_SIZE = (48, 48)  # (112, 112)
MAX_TEXT_LEN = 30
mp4_root_path = "/mnt/Work/Dataset/MERC-2020"

# tf.compat.v1.disable_eager_execution()

def single_file_process(sample_info, part='train', get_mel=False, get_face_feat=False, get_trill=False):
    file_id = sample_info['file_id']
    video_path = sample_info['video']  # .replace('XProject', 'Project')

    # Video processing #'vgg16', 'resnet50',
    # video_time = time.time()
    face_exts = ['senet50'] if get_face_feat else None
    face_info = FaceInfo(feature_extract=face_exts, is_show=False).read_video(video_path=video_path,
                                                                              out_shape=FACE_SIZE, num_skip=5,
                                                                              resize_input=0.25)

    # Audio processing
    audio, sr = librosa.load(video_path)
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
    npz = np.load('{}/2020-1/{}/{}.npz'.format(mp4_root_path, part, f'{file_id:05}'))  # Contain word embedding
    word_level_embedding_vector = npz['word_embed']

    sample_info.update({'video': face_info, 'audio': audio, 'text': word_level_embedding_vector})
    # sample_info.update({'audio': audio, 'text': word_level_embedding_vector})
    return sample_info


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(*arg):
    feature_str = dict()
    features_name_tf = ['file_id', 'face_emo', 'speech_emo', 'int_emo', 'vgg16', 'resnet50', 'senet50', 'audio', 'text']
    for idx in range(len(features_name_tf)):
        serialized_np = tf.io.serialize_tensor(arg[idx])
        feature_str[features_name_tf[idx]] = _bytes_feature(serialized_np)

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_str))
    return example_proto.SerializeToString()


def tf_serialize_example(feature_dict):
    v_list = []
    for k, v in feature_dict.items():
        v_list.append(v)

    tf_string = tf.py_function(serialize_example, v_list, tf.string)
    return tf.reshape(tf_string, ())


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
    num_pad = out_mel_length - target_mel_length
    st_crop = tf.where(num_pad <= 0, 0, tf.cast(num_pad / 2, tf.int32))
    ed_crop = tf.where(num_pad <= 0, out_mel_length, st_crop + target_mel_length)

    ret = tf.pad(db_mel_spectrogram[st_crop: ed_crop, :], tf.constant([[0, target_mel_length], [0, 0]]))[
          :target_mel_length, :]

    return ret


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


def _parse_function(element, out_dict=None, audio_length=4, video_frames=24, add_lb_in=False):
    """

    :param element:
    :param out_dict:
    :param audio_length: length of audio in second
    :param video_frames: number of frames to be used
    :return:
    """
    str_features = tf.io.FixedLenFeature([], tf.string)
    emo_dict = ['hap', 'sad', 'ang', 'sur', 'dis', 'fea', 'neu', 'ndf']
    parse_dict = {'file_id': str_features, 'face_emo': str_features, 'speech_emo': str_features,
                  'int_emo': str_features, 'video': str_features, 'audio': str_features, 'text': str_features}
    example_mess = tf.io.parse_single_example(element, parse_dict)

    if out_dict is None:
        audio = tf.io.parse_tensor(example_mess['audio'], tf.float32)[:10]
        return audio
    out_parser = dict()
    out_label_parser = dict()
    for ky in parse_dict.keys():
        if ky in out_dict or (ky == 'speech_emo' and 'text_emo' in out_dict and 'int_emo' not in out_dict):
            if ky == 'file_id':
                out_parser[ky] = tf.io.parse_tensor(example_mess[ky], tf.int32)
                # tf.print(out_parser[ky])
            elif 'emo' in ky:
                cur_emo = tf.io.parse_tensor(example_mess[ky], tf.string)
                one_hot = []
                for emo in emo_dict:
                    if emo == 'ndf':
                        break
                    else:
                        one_hot.append(tf.strings.regex_full_match(cur_emo, emo))

                out_label_parser[ky] = tf.cast(tf.stack(one_hot), tf.float32)
                out_label_parser[ky].set_shape((7,))

                if add_lb_in:
                    out_parser['{}_lb'.format(ky)] = out_label_parser[ky]

            elif ky == 'video':
                read_video = tf.io.parse_tensor(example_mess[ky], tf.uint8)

                st_crop_vd = tf.cast((tf.size(read_video) / (FACE_SIZE[0] * FACE_SIZE[1] * 3) - video_frames) / 2, tf.int32)
                # tf.print(tf.shape(read_video), st_crop_vd, st_crop_vd + video_frames)

                out_parser[ky] = tf.cast(read_video[st_crop_vd: st_crop_vd + video_frames, :, :, :], tf.float32)

                # out_parser[ky] = read_video
                out_parser[ky].set_shape((video_frames, FACE_SIZE[0], FACE_SIZE[1], 3))

            elif ky == 'audio':
                read_audio = tf.io.parse_tensor(example_mess[ky], tf.float32)
                out_parser[ky] = preprocess_audio_signal(read_audio, remove_noise=False, audio_length=audio_length)

            elif ky == 'text':
                read_text = tf.io.parse_tensor(example_mess[ky], tf.float32)

                out_parser[ky] = tf.pad(read_text, tf.constant([[0, MAX_TEXT_LEN], [0, 0]]))[:MAX_TEXT_LEN, :]

    if 'text' in out_dict and 'int_emo' not in out_dict:
        if 'audio' in out_dict:
            out_label_parser['text_emo'] = out_label_parser['speech_emo']
        else:
            out_label_parser['text_emo'] = out_label_parser.pop('speech_emo')

    if 'int_emo' in out_dict and add_lb_in:
        out_parser['text_emo_lb'] = out_parser['speech_emo_lb']

    if add_lb_in:
        out_label_parser.pop('face_emo', None)
        out_label_parser.pop('text_emo', None)
        out_label_parser.pop('speech_emo', None)

    return out_parser, out_label_parser


def to_tfrecords(sample_info, part):
    """

    :param sample_info:
    :param part:
    :return:
    """
    # print('Eager mode: ', tf.compat.v1.executing_eagerly())
    data_folder = pathlib.Path('{}/2020-1/tfrecord/{}'.format(mp4_root_path, part))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data generator')
    parser.add_argument('--start_from', type=float, default=None, required=True, help='Start from')
    args = parser.parse_args()
    print('Start from: ', args.start_from)

    # parts = ['train', 'val', 'test1', 'test2']
    # parts = ['test1', 'test2']
    parts = ['test3']
    utils.set_gpu_growth_or_cpu(use_cpu=False)
    for part in parts:
        list_emo = []
        csv_integrated = pd.read_csv('{}/2020-1/{}.csv'.format(mp4_root_path, part)).values
        if 'test' not in part:
            csv_face = pd.read_csv('{}/2020-1/{}_face.csv'.format(mp4_root_path, part)).values
            csv_speech = pd.read_csv('{}/2020-1/{}_speech.csv'.format(mp4_root_path, part)).values
        data_folder = pathlib.Path('{}/2020-1/{}'.format(mp4_root_path, part))
        list_mp4 = sorted(list(data_folder.glob('*.mp4')))
        print('Part: ', part)
        num_samples = csv_integrated.shape[0] if 'test' not in part else len(list_mp4)

        for idx in tqdm.tqdm(range(int(num_samples * args.start_from), num_samples)):
            mp4_name = str(list_mp4[idx])
            # Check id, and emo

            mp4_info = mp4_name.split('/')[-1][:-4].split('-')

            int_id = int(mp4_info[0])
            if os.path.isfile('{}/2020-1/tfrecord/{}/{}.tfrecord'.format(mp4_root_path, part, int_id)):
                check_tf = tf.data.TFRecordDataset('{}/2020-1/tfrecord/{}/{}.tfrecord'.format(mp4_root_path, part, int_id)).map(
                    _parse_function)
                ok = True
                try:
                    for dm in check_tf:
                        ok = True
                except Exception as e:
                    ok = False
                    print('{}/2020-1/tfrecord/{}/{}.tfrecord'.format(mp4_root_path, part, int_id))
                if ok:
                    continue
            if 'test' not in part:
                face_id, face_emo = csv_face[idx, :]
                speech_id, speech_emo = csv_speech[idx, :]
                int_id, int_emo = csv_integrated[idx, :]
            else:
                face_emo = 'ndf'
                speech_emo = 'ndf'
                int_emo = 'ndf'

            st = time.time()
            if 'test' in part or (
                    'test' not in part and mp4_info[0] == f'{int_id:05}' and mp4_info[0] == f'{face_id:05}' and
                    mp4_info[0] == f'{face_id:05}'):
                if 'test' in part or (int_emo == mp4_info[6] and face_emo == mp4_info[7] and speech_emo == mp4_info[8]):
                    sample_info = {'file_id': int_id, 'video': mp4_name, 'face_emo': face_emo, 'speech_emo': speech_emo,
                                   'int_emo': int_emo}
                    sample_info_feat = single_file_process(sample_info, part, get_face_feat=GET_FACE_FEAT,
                                                           get_trill=False)
                    # print(sample_info_feat.keys())
                    to_tfrecords(sample_info_feat, part)

                    # list_emo.append([int_emo, face_emo, speech_emo])
                else:
                    print('Label does not match')
            else:
                print('ID does not match ', mp4_info[0], f'{int_id:05}')

            # print('Total time: ', time.time() - st)
            # sys.exit(0)
        # list_emo = np.array(list_emo)
        # print(part, ' Integrated', np.unique(list_emo[:, 0], return_counts=True))
        # print(part, ' Face', np.unique(list_emo[:, 1], return_counts=True))
        # print(part, ' Speech', np.unique(list_emo[:, 2], return_counts=True))
    pass
