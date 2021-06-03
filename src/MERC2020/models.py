import tensorflow as tf

if tf.__version__ < '2.3.0':
    from tensorflow.python.keras import layers
else:
    from tensorflow.python.keras.layers import VersionAwareLayers

    layers = VersionAwareLayers()

from tools.tcn import TCN
import tensorflow_hub as hub
from data_generator import DEFAULT_SR, NUM_MELS
from tools import autopool
from tensorflow_addons.layers.netvlad import NetVLAD
from tools import model_utils
from tensorflow.python.keras.engine import training
from tfkeras_vggface import VGGFace
from tensorflow.keras.layers.experimental import preprocessing as tfkp
import configparser
from data_generator import FACE_SIZE
from utils import MultiModalLoss
from tools.model_utils import get_attention_adjusted


def parse_config(feature_type):
    config = configparser.ConfigParser()
    config.read('models_config.cfg')
    current_dict = config[feature_type]

    attention_bool = current_dict.getboolean('attention_pool')
    temporal_layers = [int(x) for x in current_dict['temporal_layers'].split(',')]
    act = current_dict['activation']
    ret = [attention_bool, temporal_layers, act]
    if feature_type == 'video':
        base_name = current_dict['model']
        if base_name == 'None':
            base_name = None
        ret.append(base_name)
    elif feature_type == 'text':
        max_len = int(current_dict['MAX_LEN'])
        ret.append(max_len)

    dropout = float(current_dict['dropout'])
    ret.append(dropout)
    return ret


def get_tcn_layers(x, temporal_layers, attention_pool=False):
    """

    :param x:
    :param temporal_layers:
    :param attention_pool:
    :return:
    """
    for idx in range(len(temporal_layers)):
        return_seq = False if (idx == len(temporal_layers) - 1 and not attention_pool) else True
        x = TCN(nb_filters=temporal_layers[idx], kernel_size=2, dilations=[1, 2, 4, 8], nb_stacks=1,
                use_skip_connections=True, use_layer_norm=True, return_sequences=return_seq)(x)
    return x


def audio_model(args, num_classes=7):
    """Make a model."""
    prefix = 'audio_'
    input_length = args.audio_time * 100

    attention_pool, temporal_layers, act, dropout = parse_config('audio')

    augmentation = None
    base_model = model_utils.AFCNet(width_coefficient=1., depth_coefficient=1., model_name=prefix + 'AFCNet_base',
                                    input_shape=(input_length, NUM_MELS, 1), activation='relu', pooling=None,
                                    augmentation=augmentation, rescale=False, prefix=prefix)

    target_shape = (base_model.output_shape[1], base_model.output_shape[2] * base_model.output_shape[3])
    x = base_model.output

    x = layers.Reshape(target_shape)(x)

    x = get_tcn_layers(x, temporal_layers, attention_pool)

    # Output of TCN: batch_size x timesteps x nb_filters or batch_size x nb_filters
    if attention_pool:
        x = get_attention_adjusted(x, target_shape[0], prefix, dropout=dropout, add_noise=0.0)
    else:
        x = layers.GaussianDropout(dropout)(x)

    x = layers.Dense(units=num_classes, activation=act, name='speech_emo')(x)

    model = training.Model(base_model.input, x, name=prefix + 'ACFNet')
    # tf.keras.utils.plot_model(model, '{}/check_base_audio.png'.format(args.dir), show_shapes=True, show_layer_names=True)
    return model


class VideoPreProcessing(layers.Layer):
    def __init__(self, out_shape, mean=(0., 0., 0.), stddev=(1., 1., 1.), inverse_channels=False, augmentation=None,
                 name="video_pre_processing"):
        super(VideoPreProcessing, self).__init__(name=name)
        self.out_shape = out_shape + (3,)
        self.mean = tf.constant(mean, dtype=tf.float32)
        self.stddev = tf.constant(stddev, dtype=tf.float32)
        self.inverse_channels = inverse_channels
        self.augmentation = augmentation

        self.config = {'out_shape': out_shape, 'mean': mean, 'stddev': stddev, 'inverse_channels': inverse_channels,
                       'augmentation': augmentation, 'name': name}

    def call(self, inputs):
        res = tf.reshape(inputs, (-1,) + self.out_shape)
        if self.augmentation is not None:
            res = self.augmentation(res)

        if self.inverse_channels:
            # Input is RGB, convert to BGR
            res = res[..., ::-1]

        return (res - self.mean) / self.stddev

    def get_config(self):
        return self.config


class VideoPostProcessing(layers.Layer):
    def __init__(self, video_frames, num_feat, down_feat=0., name="video_post_processing"):
        super(VideoPostProcessing, self).__init__(name=name)
        self.video_frames = video_frames
        self.num_feat = int(num_feat * down_feat) if 0. < down_feat < 1. else num_feat

        if 0. < down_feat < 1.:
            self.dense_down = layers.Dense(units=int(num_feat * down_feat), activation='relu',
                                           kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        else:
            self.dense_down = None

        self.config = {'video_frames': video_frames, 'num_feat': num_feat, 'down_feat': down_feat, 'name': name}

    def call(self, inputs):
        out_shape = (-1, self.video_frames, self.num_feat)
        if self.dense_down is not None:
            res = self.dense_down(inputs)

        else:
            res = inputs
        res = tf.reshape(res, out_shape)
        return res

    def get_config(self):
        return self.config


def video_model(args, num_classes=7):
    """ Make a model."""
    prefix = 'video_'
    img_size = FACE_SIZE
    img_input = layers.Input(shape=(args.video_frames,) + img_size + (3,), name=prefix[:-1])

    attention_pool, temporal_layers, act, model, dropout = parse_config('video')

    if model in ['senet50', 'resnet50']:
        mean = [91.4953, 103.8827, 131.0912]
        stddev = [1., 1., 1.]
        rescale = False
        inverse_channels = True
    elif model == 'vgg16':
        mean = [93.5940, 104.7624, 129.1863]
        stddev = [1., 1., 1.]
        rescale = False
        inverse_channels = True
    else:
        mean = [0., 0., 0.]
        stddev = [1., 1., 1.]
        rescale = True
        inverse_channels = False

    augmentation = tf.keras.Sequential([tfkp.RandomContrast(factor=0.5, seed=args.seed),
                                        tfkp.RandomTranslation(height_factor=0.2, width_factor=0.2, seed=args.seed),
                                        tfkp.RandomRotation(factor=0.2, fill_mode='constant', seed=args.seed)])

    post_img_input = VideoPreProcessing(img_size, mean=mean, stddev=stddev, inverse_channels=inverse_channels,
                                        augmentation=augmentation)(img_input)
    if model is None:
        base_model = model_utils.AFCNet(width_coefficient=1., depth_coefficient=1., model_name=prefix + 'AFCNet_base',
                                        input_tensor=post_img_input, activation='relu', pooling='max',
                                        augmentation=None, rescale=rescale, prefix=prefix, use_low_feat=False)
    else:
        base_model = VGGFace(include_top=False, input_tensor=post_img_input, model=model, pooling='avg')

    x = VideoPostProcessing(args.video_frames, base_model.output_shape[1], down_feat=0.)(base_model.output)

    x = get_tcn_layers(x, temporal_layers, attention_pool)

    # Output of TCN: batch_size x timesteps x nb_features or batch_size x nb_features
    if attention_pool:
        x = get_attention_adjusted(x, args.video_frames, prefix, dropout=dropout, add_noise=0.0)
    else:
        x = layers.GaussianDropout(dropout)(x)

    x = layers.Dense(units=num_classes, activation=act, name='face_emo')(x)

    model = training.Model(base_model.input, x, name=prefix + 'ACFNet')
    # tf.keras.utils.plot_model(model, 'check_base.png', show_shapes=True, show_layer_names=True)
    return model


def text_model(args, num_classes=7):
    """ Create text model"""
    prefix = 'text_'

    attention_pool, temporal_layers, act, MAX_LEN, dropout = parse_config('text')

    text_input = layers.Input(shape=(MAX_LEN, 200), name=prefix[:-1])
    x = layers.Masking(mask_value=0.)(text_input)

    x = get_tcn_layers(x, temporal_layers, attention_pool)

    if attention_pool:
        x = get_attention_adjusted(x, MAX_LEN, prefix, dropout=dropout, add_noise=0.0)
    else:
        x = layers.GaussianDropout(dropout)(x)

    x = layers.Dense(units=num_classes, activation=act, name='text_emo')(x)

    model = training.Model(text_input, x, name=prefix + 'ACFNet')
    # tf.keras.utils.plot_model(model, 'check_base.png', show_shapes=True, show_layer_names=True)
    return model


def multimodal_model(args, num_classes=7):
    """
    Create multimodal
    :param args:
    :param num_classes:
    :param freeze:
    :param multi_outputs:
    :param act:
    :return:
    """
    base_audio_model = audio_model(args)
    base_video_model = video_model(args)
    base_text_model = text_model(args)

    config = configparser.ConfigParser()
    config.read('models_config.cfg')
    act = config['multimodal']['activation']
    dropout = float(config['multimodal']['dropout'])
    freeze = config['multimodal'].getboolean('freeze')
    multi_outputs = config['multimodal'].getboolean('multi_outputs')
    use_ckpt = config['multimodal'].getboolean('use_ckpt')

    if use_ckpt and freeze:
        try:
            base_audio_model.load_weights('{}/{}_best_checkpoint.h5'.format(args.ckpt, 'audio'))
            base_video_model.load_weights('{}/{}_best_checkpoint.h5'.format(args.ckpt, 'video'))
            base_text_model.load_weights('{}/{}_best_checkpoint.h5'.format(args.ckpt, 'text'))
        except Exception as e:
            print(e)
            return -1

        base_audio_model.trainable = False
        base_video_model.trainable = False
        base_text_model.trainable = False
    # else:
    #     base_audio_model.trainable = True
    #     base_video_model.trainable = True
    #     base_text_model.trainable = True

    audio_feat = base_audio_model.layers[-2].output
    video_feat = base_video_model.layers[-2].output
    text_feat = base_text_model.layers[-2].output

    audio_feat = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                              name='audio_feat')(audio_feat)
    video_feat = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                              name='video_feat')(video_feat)
    text_feat = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                             name='text_feat')(text_feat)
    mm_feat = tf.stack([audio_feat, video_feat, text_feat], axis=2)  # Batch size x 32 x 3
    mm_feat = get_attention_adjusted(mm_feat, 3, prefix='mm', permute=False)
    mm_feat = layers.Dropout(dropout)(mm_feat)

    # print(multi_outputs)
    feat_dict = {'audio': base_audio_model.input, 'video': base_video_model.input, 'text': base_text_model.input}
    if multi_outputs:
        mm_output = layers.Dense(num_classes, activation=act)(mm_feat)

        in_out = []
        targets = ['speech_emo', 'face_emo', 'text_emo', 'int_emo']
        for idx in range(4):
            # 7 is the number of classes
            feat_dict.update(
                {'{}_lb'.format(targets[idx]): tf.keras.layers.Input(shape=(7,), name='{}_lb'.format(targets[idx]))})
            in_out.append(feat_dict['{}_lb'.format(targets[idx])])

        outs = [base_audio_model.output, base_video_model.output, base_text_model.output, mm_output]
        out_mtl = MultiModalLoss(num_outputs=4, loss_function=tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                                                                      label_smoothing=0.0),
                                 trainable=True, name='int_emo')(in_out + outs)

        outputs = out_mtl
    else:
        mm_output = layers.Dense(num_classes, activation=act, name='int_emo')(mm_feat)
        outputs = {'int_emo': mm_output}

    model = training.Model(feat_dict, outputs, name='MultiModal')
    # tf.keras.utils.plot_model(model, 'check_mm.png', show_shapes=True, show_layer_names=True)
    return model
