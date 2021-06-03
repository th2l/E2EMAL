"""
Created by Thong

Face extractor from video and face feature extraction
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import cv2
from mtcnn import MTCNN
import utils
from collections import deque
from multiprocessing.pool import ThreadPool
import time, sys
from threading import Thread
import numpy as np
from tfkeras_vggface.vggface import VGGFace
from tfkeras_vggface import utils as tfvgg_utils
import utils

class FaceInfo():
    def __init__(self, feature_extract=None, is_show=False):
        self.results = deque()
        self.inps = deque()

        self.face_thread = Thread(target=self.face_detector, args=())
        self.face_thread.daemon = True
        self.face_thread.start()

        self.stop = False
        self.is_show = is_show
        self.feature_extract = feature_extract
        self.stop_thread = False

    def face_detector(self):
        tf.compat.v1.disable_eager_execution()
        utils.set_gpu_growth_or_cpu()
        detector = MTCNN()

        while len(self.inps) > 0 or not self.stop:
            # print(len(self.inps), len(self.results))
            if len(self.inps) > 0:
                cur_inps, previous = self.inps.popleft()

                if previous is not None:
                    if self.is_show:
                        cv2.rectangle(cur_inps, previous[:2], previous[2:], (0, 155, 255), thickness=5)
                    self.results.append((cur_inps, previous))
                else:
                    st = time.time()
                    results = detector.detect_faces(cur_inps)
                    # print('Total time face det : ', time.time() - st)
                    img_h, img_w, _ = cur_inps.shape
                    max_face = -1
                    if len(results) > 0:
                        for candidate in results:
                            if candidate['confidence'] < 0.65:
                                continue
                            box = candidate['box']  # x, y, w, h
                            xl = max(box[0], 0)
                            yt = max(box[1], 0)
                            xr = min(img_w, box[0] + box[2])
                            yb = min(img_h, box[1] + box[3])
                            c_area = (xr - xl) * (yb - yt)
                            if self.is_show:
                                cv2.rectangle(cur_inps, (xl, yt), (xr, yb), (0, 155, 255), thickness=5)
                            if max_face == -1:
                                max_face = (xl, yt, xr, yb, c_area)
                            elif max_face[-1] < c_area:
                                max_face = (xl, yt, xr, yb, c_area)
                        self.results.append((cur_inps, max_face))
                    else:
                        self.results.append((cur_inps, None))
        self.stop_thread = True


    def resize_face(self, face_img, target_shape=(224, 224)):
        """

        :param target_shape:
        :return:
        """
        if target_shape[0] != target_shape[1]:
            raise ValueError("Do not support non-square shape")
        target_size = target_shape[0]

        ratio = target_size / max(face_img.shape[0], face_img.shape[1])
        face_img = cv2.resize(face_img, dsize=None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
        left_pad = (target_size - face_img.shape[1]) // 2
        right_pad = target_size - (left_pad + face_img.shape[1])
        top_pad = (target_size - face_img.shape[0]) // 2
        bottom_pad = target_size - (top_pad + face_img.shape[0])

        face_img = cv2.copyMakeBorder(face_img, top=top_pad, bottom=bottom_pad, left=left_pad, right=right_pad,
                                      borderType=cv2.BORDER_CONSTANT, value=0)
        return face_img

    def read_video(self, video_path=None, num_skip=0, out_shape=(224, 224), resize_input=0.5):
        """

        :param video_path:
        :param num_skip:
        :return: Out faces in RGB
        """
        face_regions = []
        if video_path is not None:
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(0)

        # Check if camera opened successfully

        if not cap.isOpened():
            print("Error opening video stream or file")
            return
        count_num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        while num_skip > 0 and count_num_frames // num_skip <= 40:
            num_skip = num_skip - 1

        count_frames = 0
        max_face = []

        # st_h = time.time()
        while True:
            while len(self.results) > 0:
                out, max_face = self.results.popleft()

                if max_face is None:
                    continue
                out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                if self.is_show:
                    cv2.imshow('Face detection', out)

                # Crop face, out face in RGB
                cur_face = self.resize_face(out[max_face[1]: max_face[3], max_face[0]: max_face[2], :],
                                            target_shape=out_shape)
                face_regions.append(cur_face)

            # Capture frame by frame
            ret, frame = cap.read()
            if ret:
                count_frames += 1
                inp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                inp = cv2.resize(inp, None, fx=resize_input, fy=resize_input)
                # print(inp.copy().shape)
                if num_skip == 0 or (num_skip > 0 and count_frames % (num_skip + 1) == 0):
                    # Do face detector
                    # print('Append to inps')
                    self.inps.append((inp.copy(), None))
                else:
                    # Use previous results
                    # print('Use previous results')
                    # self.inps.append((inp.copy(), max_face))
                    continue
            else:
                self.stop = True

            if self.stop_thread and len(self.results) == 0:
                break
            if self.is_show:
                ch = cv2.waitKey(1)
                if ch == 27:
                    break

        cap.release()
        if self.is_show:
            cv2.destroyWindow('Face detection')
        # print('Time read process video: ', time.time()-st_h)
        if len(face_regions) == 0:
            face_regions = np.zeros(shape=(32, ) + out_shape + (3,), dtype=np.uint8)
        else:
            face_regions = np.array(face_regions)
        face_feats = dict()
        # print(face_regions.shape, count_frames)

        if self.feature_extract is not None:
            for fext in self.feature_extract:
                include_top = (fext == 'vgg16')
                vggface = VGGFace(model=fext, include_top=include_top, input_shape=out_shape + (3, ), pooling='avg')
                if fext == 'vgg16':
                    out = vggface.get_layer('fc6').output
                else:
                    # out = vggface.get_layer('avg_pool').output
                    out = vggface.output
                fext_model = tf.keras.models.Model(vggface.input, out)
                x = tfvgg_utils.preprocess_input(face_regions*1.0, version=1 if fext=='vgg16' else 2)
                preds = fext_model.predict(x, batch_size=128)
                face_feats[fext] = preds

            tf.keras.backend.clear_session()
            return face_feats
        else:
            return face_regions


if __name__ == '__main__':
    utils.set_gpu_growth_or_cpu(use_cpu=False)
    obj = FaceInfo()
    # obj.read_video(video_path=None)
    obj.read_video(video_path="./2020-1/train19180-4-092-m-35-075-sad-sad-neu.mp4")