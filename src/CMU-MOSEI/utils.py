import sys

import tensorflow as tf
import math
import numpy as np
import random, time
import tabulate
import copy
from functools import partial
import tensorflow_addons as tfa

def set_gpu_growth_or_cpu(use_cpu=False, write_info=False):

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        if use_cpu:
            if write_info:
                print("Use CPU")
            tf.config.set_visible_devices(gpus[1:], 'GPU')
        else:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                if write_info:
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs, ", len(logical_gpus), " Logical GPUs")
                    print('Use GPU')
            except RuntimeError as e:
                print(e)
    else:
        print("Running CPU, please check GPU drivers, CUDA, ...")

    tf.get_logger().setLevel('INFO')


def set_seed(seed, reset_session=False):
    if reset_session:
        tf.keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class VerboseFitCallBack(tf.keras.callbacks.Callback):
    def __init__(self, print_lr=False):
        super(VerboseFitCallBack).__init__()
        self.columns = None
        self.st_time = 0
        self.print_lr = print_lr

    def on_epoch_begin(self, epoch, logs=None):
        self.st_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        cus_logs = copy.deepcopy(logs)
        cus_logs.pop('batch', None)
        cus_logs.pop('size', None)

        current_header = list(cus_logs.keys())
        if 'lr' in current_header:
            lr_index = current_header.index('lr')
        else:
            lr_index = len(current_header)

        if self.columns is None:

            self.columns = current_header[:lr_index] + current_header[lr_index + 1:] + ['time']
            if self.print_lr and tf.executing_eagerly():
                self.columns = ['ep', 'lr'] + self.columns
            else:
                self.columns = ['ep',] + self.columns
            # for col_index in range(len(self.columns)):
            #     if len(self.columns[col_index]) > 10:
            #         self.columns[col_index] = self.columns[col_index][:10]
        logs_values = list(cus_logs.values())

        if self.print_lr and tf.executing_eagerly():
            # Get Learning rate
            current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
            try:
                current_step = tf.cast(self.model.optimizer.iterations, tf.float32)
                current_lr = float(current_lr(current_step))
                # current_lr = tf.cast(current_lr(current_step), tf.float32)
            except:
                current_lr = float(current_lr)

        time_ep = time.time() - self.st_time
        if self.print_lr and tf.executing_eagerly():
            current_values = [epoch + 1, current_lr] + logs_values[:lr_index] + logs_values[lr_index + 1:] + [time_ep]
        else:
            current_values = [epoch + 1,] + logs_values[:lr_index] + logs_values[lr_index + 1:] + [time_ep]

        table = tabulate.tabulate([current_values], self.columns, tablefmt='simple', floatfmt='10.6g')
        if epoch % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)


class CusLRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, min_lr, lr_start_warmup=0., warmup_steps=10, num_constant=0, T_max=20, num_half_cycle=1.,
                 name=None):
        super(CusLRScheduler, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.num_constant = num_constant
        self.T_max = T_max
        self.lr_start_warmup = lr_start_warmup
        self.min_lr = min_lr
        self.num_half_cycle = num_half_cycle
        self.name = name
        pass

    def __call__(self, step):
        with tf.name_scope(self.name or "CusLRScheduler") as name:
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype

            min_lr = tf.cast(self.min_lr, dtype)
            lr_start_warmup = tf.cast(self.lr_start_warmup, dtype)

            step_cf = tf.cast(step, dtype)
            wm_steps = tf.cast(self.warmup_steps, dtype=dtype)

            warmup_ratio = tf.where(tf.less_equal(step_cf, wm_steps), step_cf / wm_steps, 0.0)
            use_warmup_lr = tf.where(tf.less_equal(step_cf, wm_steps), 1.0, 0.0)
            warmup_lr = use_warmup_lr * (lr_start_warmup + warmup_ratio * (initial_learning_rate - lr_start_warmup))

            num_constant = tf.cast(self.num_constant, dtype=dtype)

            constant_lr = tf.where(
                tf.logical_and(tf.less_equal(step_cf - wm_steps, num_constant), use_warmup_lr<1),
                initial_learning_rate, 0.0)

            t_max = tf.cast(self.T_max, dtype)
            use_consine_lr = tf.where(tf.logical_and(tf.less_equal(step_cf, t_max), tf.less(wm_steps + num_constant, step_cf)), 1.0, 0.0)
            pi_val = tf.cast(tf.constant(math.pi), dtype)
            num_half_cycle = tf.cast(self.num_half_cycle, dtype)
            cosine_lr = tf.where(use_consine_lr>0., min_lr + (initial_learning_rate - min_lr) * (1 + tf.cos(
                pi_val * num_half_cycle*(step_cf - wm_steps - num_constant) / (t_max - wm_steps - num_constant))) / 2, 0.)

            use_min_lr = tf.where(tf.less_equal(t_max, step_cf), min_lr, 0.0)

            return use_min_lr + cosine_lr + constant_lr + warmup_lr

    def get_config(self):
        ret_config = {'initial_learning_rate': self.initial_learning_rate,
                      'min_lr': self.min_lr,
                      'lr_start_warmup': self.lr_start_warmup,
                      'warmup_steps': self.warmup_steps,
                      'num_constant': self.num_constant,
                      'T_max': self.T_max,
                      'num_half_cycle': self.num_half_cycle,
                      'name': self.name}
        return ret_config

class MultiModalLoss(tf.keras.layers.Layer):
    """Adapted from https://github.com/yaringal/multi-task-learning-example"""
    def __init__(self, num_outputs=4, loss_function=tf.keras.losses.CategoricalCrossentropy, trainable=True, **kwargs):
        self.num_outputs = num_outputs
        self.loss_func = loss_function
        self.trainable = trainable
        self.config = {'num_outputs': num_outputs, 'loss_function': loss_function, 'trainable': trainable}
        super(MultiModalLoss, self).__init__(**kwargs)

    def build(self, input_shape=None):
        self.log_vars = []
        for idx in range(self.num_outputs):
            self.log_vars += [self.add_weight(name='log_var_{}'.format(idx), shape=(1, ), initializer=tf.keras.initializers.Constant(0.), trainable=self.trainable)]

        super(MultiModalLoss, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.num_outputs and len(ys_pred) == self.num_outputs
        loss = 0
        idx = 0
        for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
            prec = tf.exp(-log_var[0])
            loss = loss + prec * self.loss_func(y_true, y_pred)
        loss = loss + 0.5 * tf.reduce_mean(self.log_vars)

        # for y_true, y_pred in zip(ys_true, ys_pred):
        #     # prec = tf.exp(-log_var[0])
        #     # Only use mm
        #     prec = 0.0 if idx < 3 else 1.0
        #     idx += 1
        #     loss = loss + prec * self.loss_func(y_true, y_pred)

        return loss

    def call(self, inputs):
        ys_true = inputs[: self.num_outputs]
        ys_pred = inputs[self.num_outputs: ]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)

        # Only return last prediction
        return inputs[-1]

    def get_config(self):
        return self.config

class MoseiEmotionLoss(tf.keras.losses.Loss):

    def __init__(self, alpha=0.25, gamma=2.0, num_classes=6, optimize_class=-1):
        super(MoseiEmotionLoss, self).__init__()
        # self.loss_func = tfa.losses.SigmoidFocalCrossEntropy(from_logits=False, alpha=alpha, gamma=gamma, reduction=tf.keras.losses.Reduction.AUTO)
        self.loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.AUTO)
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.optimize_class = optimize_class

    def call(self, y_true, y_pred):
        # sce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        # Calculate class-weight
        # pos_count = tf.reduce_sum(y_true, axis=0)
        # sce = 0.0
        # for idx in range(self.num_classes):
        #     sce += self.loss_func(y_true=y_true[:, idx], y_pred=y_pred[:, idx])
        #
        # current_loss = sce / self.num_classes
        # current_loss = tf.reduce_mean(sce)
        class_weights = [[1/0.53, 1/0.47], [1/0.25, 1/0.75], [1/0.21, 1/0.79], [1/0.09, 1/0.91], [1/0.16, 1/0.84], [1/0.08, 1/0.92]]
        y_pred_sigmoid = tf.sigmoid(y_pred)
        sce = 0
        for idx in range(self.num_classes):
            if -1 < self.optimize_class != idx:
                continue
            idx_y_true = tf.transpose(tf.stack([y_true[:, idx], 1 - y_true[:, idx]]))
            idx_y_pred = tf.transpose(tf.stack([y_pred_sigmoid[:, idx], 1 - y_pred_sigmoid[:, idx]]))

            # sce += self.loss_func(idx_y_true, idx_y_pred)
            current_class_weight = tf.reduce_sum(tf.convert_to_tensor([class_weights[idx]]) * tf.ones_like(idx_y_true), axis=-1)
            current_bce = self.loss_func(idx_y_true, idx_y_pred, sample_weight=current_class_weight)
            sce = sce + current_bce

        current_loss = sce / self.num_classes
        return current_loss

    def get_config(self):
        return {'gamma': self.gamma, 'alpha': self.alpha, 'num_classes': self.num_classes}


class F1ScoreMetric(tf.keras.metrics.Metric):
    def __init__(self, name='f1score', num_classes=6, average=None, threshold=0.5, **kwargs):
        """ Weighted F1 score"""
        super(F1ScoreMetric, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average
        self.threshold = threshold
        self.f1_funcs = []
        for idx in range(self.num_classes):
            # Binary emotion
            self.f1_funcs.append(tfa.metrics.F1Score(num_classes=2, average=average, threshold=threshold))

    def update_state(self, y_true, logits, sample_weight=None):
        y_pred_sigmoid = tf.sigmoid(logits)

        for idx in range(self.num_classes):
            idx_y_true = tf.transpose(tf.stack([y_true[:, idx], 1 - y_true[:, idx]]))
            idx_y_pred = tf.transpose(tf.stack([y_pred_sigmoid[:, idx], 1 - y_pred_sigmoid[:, idx]]))
            self.f1_funcs[idx].update_state(y_true=idx_y_true, y_pred=idx_y_pred)

    def per_class_result(self):
        return self.result(True)

    def result(self, printout=False):
        ret = []

        for idx in range(self.num_classes):
            ret.append(self.f1_funcs[idx].result())

        if printout:
            return ret
        ret = tf.reduce_mean(ret)
        return tf.reduce_mean(ret)

    def reset_states(self):
        for idx in range(self.num_classes):
            self.f1_funcs[idx].reset_states()

    def get_config(self):
        return {'num_classes': self.num_classes, 'average': self.average, 'threshold': self.threshold}

class WeightedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='wAcc', num_classes=6, threshold=0.5, **kwargs):
        super(WeightedAccuracy, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold

        self.list_tp = self.add_weight(name='list_tp', shape=[num_classes], initializer='zeros', dtype=self.dtype)
        self.list_tn = self.add_weight(name='list_tp', shape=[num_classes], initializer='zeros', dtype=self.dtype)
        self.list_pos = self.add_weight(name='list_tp', shape=[num_classes], initializer='zeros', dtype=self.dtype)
        self.list_neg = self.add_weight(name='list_tp', shape=[num_classes], initializer='zeros', dtype=self.dtype)

    def reset_states(self):
        reset_value = tf.zeros(self.num_classes, dtype=self.dtype)
        tf.keras.backend.set_value(self.list_tp, reset_value)
        tf.keras.backend.set_value(self.list_tn, reset_value)
        tf.keras.backend.set_value(self.list_pos, reset_value)
        tf.keras.backend.set_value(self.list_neg, reset_value)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_sigmoid = tf.nn.sigmoid(y_pred)
        y_pred_val = y_pred_sigmoid > self.threshold

        y_true = tf.cast(y_true, self.dtype)
        y_pred_val = tf.cast(y_pred_val, self.dtype)

        self.list_tp.assign_add(tf.reduce_sum(y_true * y_pred_val, axis=0))
        self.list_tn.assign_add(tf.reduce_sum((1-y_true) * (1-y_pred_val), axis=0))
        self.list_pos.assign_add(tf.reduce_sum(y_true, axis=0))
        self.list_neg.assign_add(tf.reduce_sum(1-y_true, axis=0))

    def per_class_result(self):
        return self.result(True)

    def result(self, printout=False):
        ret_wacc = tf.divide(self.list_tp * (self.list_neg / self.list_pos) + self.list_tn, 2*self.list_neg)
        ret = tf.where(tf.equal(self.list_neg, 0), self.list_tp / self.list_pos, ret_wacc)
        if printout:
            return ret
        return tf.reduce_mean(ret)


