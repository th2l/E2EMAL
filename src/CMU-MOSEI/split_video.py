import socket

from moviepy.editor import VideoFileClip, vfx
# from moviepy.video import fx as vfx
from joblib import Parallel, delayed
import pandas as pd
import h5py
from foldSplitIDs import data_partitions_ID
import os
import numpy as np
import argparse


def label_parsing(arrs):
    """
    https://github.com/A2Zadeh/CMU-MultimodalSDK/tree/master/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI
    :param arrs: [sentiment, happy, sad, anger, surprise, disgust, fear]
    :return: s7 (7-class sentiment), s2o (binary sentiment prior 2019), s2n (binary sentiment after 2019), emo_arrs (binary emotions)
    """
    # 7-class sentiment
    a = arrs[0]  # Sentiment
    if a < -2:
        s7 = -3
    elif -2 <= a < -1:
        s7 = -2
    elif -1 <= a < 0:
        s7 = -1
    elif 0 <= a <= 0:
        s7 = 0
    elif 0 < a <= 1:
        s7 = 1
    elif 1 < a <= 2:
        s7 = 2
    else:  # if a > 2:
        s7 = 3

    s2o = 0 if a < 0 else 1  # 0 - negative, 1 - non negative, prior 2019
    # 0 - negative, 1 - positive, after 2019
    if a < 0:
        s2n = 0
    elif a > 0:
        s2n = 1
    else:
        s2n = -1

    emo_arrs = []
    for idx in range(1, 7):
        emo_arrs.append(0) if arrs[idx] == 0 else emo_arrs.append(1)

    # s7 is in [-3,-2,-1,0,1,2,3] => we plus by 3 for easy handling later => [0,1,2,3,4,5,6]
    return [s7 + 3, s2o, s2n] + emo_arrs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split video')
    parser.add_argument('--start_from', type=int, default=0, required=False)
    args = parser.parse_args()

    root_path = "/mnt/Data/Dataset/CMU_MOSEI/"
    time_offset = pd.read_csv("{}Check_data.csv".format(root_path))

    re_check_vd = ['-iRBcNs9oI8', '-ZgjBOA1Yhw', '-lzEya4AM_4', '-wny0OAz3g8', '-3nNcZdcdvU', '-aNfi7CP8vM',
                   '-vxjVxOeScU', '-aqamKhZ1Ec', '-3g5yACwYnA', '-tANM6ETl_M', '-egA8-b7-3M',
                   '-tPCytz4rww', '-lqc32Zpr7M', '-THoVjtIkeU', '-wMB_hJL-3o', '-a55Q6RWvTA', '-UuX1xuaiiE',
                   '-mJ2ud6oKI8', '-mqbVkbCndg', '-WXXTNIJcVM', '-HwX2H8Z4hY', '-t217m2on-s',
                   '-dxfTGcXJoc', '--qXJuDtHPw', '-uywlfIYOS8', '-UacrmKiTn4', '-I_e4mIh0yE', '-qDkUB0GgYY',
                   '-hnBHBN8p5A', '-571d8cVauQ', '-rxZxtG0xmY', '-6rXp3zJ3kc', '-cEhr0cQcDM',
                   '-s9qJ7ATP7w', '-RfYyzHpjk4', '-HeZS2-Prhc', '-9y-fZ3swSY', '-UUCSKoHeMA', '-MeTTeMJBNc',
                   '-yRb-Jum7EQ', '-AUZQgSxyPQ', '-ri04Z7vwnc']
    flabel = h5py.File('{}Raw/All Labels.csd'.format(root_path), 'r')
    fglove_vectors = h5py.File('{}Raw/glove_vectors.csd'.format(root_path), 'r')

    list_keys = list(flabel['All Labels']['data'].keys())
    list_keys_woIndex = np.array([x.split('[')[0] for x in list_keys])

    os.makedirs('{}dataset'.format(root_path), exist_ok=True)
    index_count = 0
    for split in data_partitions_ID.keys():
        # split will be train, val, test
        os.makedirs('{}dataset/{}'.format(root_path, split), exist_ok=True)
        print(split)
        current_fold = data_partitions_ID[split]
        current_fold_labels = []
        for kid in current_fold:
            index_count += 1

            # if kid not in re_check_vd:
            #     continue
            if index_count % 1000 == 0:
                print(index_count, '/', len(current_fold))
            list_index = np.argwhere(list_keys_woIndex == kid).flatten()
            if len(list_index) == 0:
                continue

            # current_clip = VideoFileClip('{}/Raw/Videos/Full/Combined/{}.mp4'.format(root_path, kid))
            # clip_duration = current_clip.duration
            # if min(current_clip.w, current_clip.h) >= 720:
            #     current_clip = current_clip.fx(vfx.resize, 0.5)
            for idx in list_index:
                current_key = list_keys[idx]
                # if os.path.isfile('{}dataset/{}/{}.mp4'.format(root_path, split, current_key.replace('[', '_')[:-1])):
                #     continue
                arr_labels = flabel['All Labels']['data'][current_key]['features'][()][0]
                arr_timestamp = flabel['All Labels']['data'][current_key]['intervals'][()][0]
                # arr_glove_vectors = np.mean(fglove_vectors['glove_vectors']['data'][current_key]['features'][()],
                #                             axis=0)
                arr_glove_vectors = fglove_vectors['glove_vectors']['data'][current_key]['features'][()]

                arr_glove_vectors_timestamp = fglove_vectors['glove_vectors']['data'][current_key]['intervals'][()]

                # vd_st = max(arr_timestamp[0], 0)
                # vd_ed = min(arr_timestamp[1], clip_duration)
                # current_subclip = current_clip.subclip(vd_st, vd_ed)

                # try:
                #     sclip_write_name = '{}dataset/{}/{}.mp4'.format(root_path, split,
                #                                                     current_key.replace('[', '_')[:-1])
                #     if current_key[0] == '-':
                #         tmp_name = '{}dataset/{}/{}.mp4'.format(root_path, split,
                #                                                     'X' + current_key.replace('[', '_')[1:-1])
                #     else:
                #         tmp_name = sclip_write_name
                #     current_subclip.write_videofile(tmp_name, verbose=False, logger=None)
                #     if tmp_name != sclip_write_name:
                #         os.rename(tmp_name, sclip_write_name)
                #
                # except BrokenPipeError as e:
                #     print('Error: ', current_key)
                # except socket.error as e:
                #     print('Error: ', current_key)
                #     continue
                np.save('{}dataset/{}/{}.npy'.format(root_path, split, current_key.replace('[', '_')[:-1]),
                        arr_glove_vectors)
                # current_labels = label_parsing(arr_labels)
                # current_fold_labels.append([current_key.replace('[', '_')[:-1]] + current_labels)

            # current_clip.close()

        # df_label = pd.DataFrame(current_fold_labels,
        #                         columns=['Name', 'S7', 'S2o', 'S2n', 'happy', 'sad', 'anger', 'surprise', 'disgust',
        #                                  'fear'])
        # df_label.to_csv('{}.csv'.format(split), index=False)

    pass
