import tensorflow.keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from Balanced_Affinity_loss import *

# Source:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py


from .config import IMAGE_ORDERING


if IMAGE_ORDERING == 'channels_first':
    pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                     "releases/download/v0.2/" \
                     "resnet50_weights_th_dim_ordering_th_kernels_notop.h5"
elif IMAGE_ORDERING == 'channels_last':
    pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                     "releases/download/v0.2/" \
                     "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

#
# def one_side_pad(x):
#     x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
#     if IMAGE_ORDERING == 'channels_first':
#         x = Lambda(lambda x: x[:, :, :-1, :-1])(x)
#     elif IMAGE_ORDERING == 'channels_last':
#         x = Lambda(lambda x: x[:, :-1, :-1, :])(x)
#     return x
#
#
# def identity_block(input_tensor, kernel_size, filters, stage, block):
#     """The identity block is the block that has no conv layer at shortcut.
#     # Arguments
#         input_tensor: input tensor
#         kernel_size: defualt 3, the kernel size of middle conv layer at
#                      main path
#         filters: list of integers, the filterss of 3 conv layer at main path
#         stage: integer, current stage label, used for generating layer names
#         block: 'a','b'..., current block label, used for generating layer names
#     # Returns
#         Output tensor for the block.
#     """
#     filters1, filters2, filters3 = filters
#
#     if IMAGE_ORDERING == 'channels_last':
#         bn_axis = 3
#     else:
#         bn_axis = 1
#
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     x = Conv2D(filters1, (1, 1), data_format=IMAGE_ORDERING,
#                name=conv_name_base + '2a')(input_tensor)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters2, kernel_size, data_format=IMAGE_ORDERING,
#                padding='same', name=conv_name_base + '2b')(x)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING,
#                name=conv_name_base + '2c')(x)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
#
#     x = layers.add([x, input_tensor])
#     x = Activation('relu')(x)
#     return x
#
#
# def conv_block(input_tensor, kernel_size, filters, stage, block,
#                strides=(2, 2)):
#     """conv_block is the block that has a conv layer at shortcut
#     # Arguments
#         input_tensor: input tensor
#         kernel_size: defualt 3, the kernel size of middle conv layer at
#                      main path
#         filters: list of integers, the filterss of 3 conv layer at main path
#         stage: integer, current stage label, used for generating layer names
#         block: 'a','b'..., current block label, used for generating layer names
#     # Returns
#         Output tensor for the block.
#     Note that from stage 3, the first conv layer at main path is with
#     strides=(2,2) and the shortcut should have strides=(2,2) as well
#     """
#     filters1, filters2, filters3 = filters
#
#     if IMAGE_ORDERING == 'channels_last':
#         bn_axis = 3
#     else:
#         bn_axis = 1
#
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     x = Conv2D(filters1, (1, 1), data_format=IMAGE_ORDERING, strides=strides,
#                name=conv_name_base + '2a')(input_tensor)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters2, kernel_size, data_format=IMAGE_ORDERING,
#                padding='same', name=conv_name_base + '2b')(x)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING,
#                name=conv_name_base + '2c')(x)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
#
#     shortcut = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING,
#                       strides=strides, name=conv_name_base + '1')(input_tensor)
#     shortcut = BatchNormalization(
#         axis=bn_axis, name=bn_name_base + '1')(shortcut)
#
#     x = layers.add([x, shortcut])
#     x = Activation('relu')(x)
#     return x
#
#
# def get_resnet50_encoder(input_height=224,  input_width=224,
#                          pretrained='imagenet',
#                          include_top=True, weights='imagenet',
#                          input_tensor=None, input_shape=None,
#                          pooling=None,
#                          classes=1000):
#
#     assert input_height % 32 == 0
#     assert input_width % 32 == 0
#
#     if IMAGE_ORDERING == 'channels_first':
#         img_input = Input(shape=(3, input_height, input_width))
#     elif IMAGE_ORDERING == 'channels_last':
#         img_input = Input(shape=(input_height, input_width, 3))
#
#     if IMAGE_ORDERING == 'channels_last':
#         bn_axis = 3
#     else:
#         bn_axis = 1
#
#     x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(img_input)
#     x = Conv2D(64, (7, 7), data_format=IMAGE_ORDERING,
#                strides=(2, 2), name='conv1')(x)
#     f1 = x
#
#     x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
#     x = Activation('relu')(x)
#     x = MaxPooling2D((3, 3), data_format=IMAGE_ORDERING, strides=(2, 2))(x)
#
#     x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
#     x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
#     x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
#     f2 = one_side_pad(x)
#
#     x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
#     f3 = x
#
#     x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
#     f4 = x
#
#     x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
#     x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
#     x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
#     f5 = x
#
#     x = AveragePooling2D(
#         (7, 7), data_format=IMAGE_ORDERING, name='avg_pool')(x)
#     # f6 = x
#
#     if pretrained == 'imagenet':
#         weights_path = tensorflow.keras.utils.get_file(
#             pretrained_url.split("/")[-1], pretrained_url)
#         Model(img_input, x).load_weights(weights_path)
#
#     return img_input, [f1, f2, f3, f4, f5]



def conv3x3(x, out_filters, strides=(1, 1)):
    x = Conv2D(out_filters, 3, padding='same', strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
    return x

def basic_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    x = conv3x3(input, out_filters, strides)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = conv3x3(x, out_filters)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x


def bottleneck_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    expansion = 4
    de_filters = int(out_filters / expansion)

    x = Conv2D(de_filters, 1, use_bias=False, kernel_initializer='he_normal')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(de_filters, 3, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(out_filters, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x


def stem_net(input):
    x = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = bottleneck_Block(x, 224, with_conv_shortcut=True)
    x = bottleneck_Block(x, 224, with_conv_shortcut=False)
    x = bottleneck_Block(x, 224, with_conv_shortcut=False)
    x = bottleneck_Block(x, 224, with_conv_shortcut=False)

    return x


def transition_layer1(x, out_filters_list=[32, 64]):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x0 = BatchNormalization(axis=3)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, strides=(2, 2),
                padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)

    return [x0, x1]


def make_branch1_0(x, out_filters=32):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x

def make_branch1_1(x, out_filters=64):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def fuse_layer1(x):
    x0_0 = x[0]
    x0_1 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = BatchNormalization(axis=3)(x0_1)
    x0_1 = UpSampling2D(size=(2, 2))(x0_1)
    x0 = add([x0_0, x0_1])

    x1_0 = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x1_0 = BatchNormalization(axis=3)(x1_0)
    x1_1 = x[1]
    x1 = add([x1_0, x1_1])
    return [x0, x1]


def transition_layer2(x, out_filters_list=[32, 64, 224]):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x0 = BatchNormalization(axis=3)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(out_filters_list[2], 3, strides=(2, 2),
                padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x2 = BatchNormalization(axis=3)(x2)
    x2 = Activation('relu')(x2)

    return [x0, x1, x2]


def make_branch2_0(x, out_filters=32):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch2_1(x, out_filters=64):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch2_2(x, out_filters=224):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def fuse_layer2(x):
    x0_0 = x[0]
    x0_1 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = BatchNormalization(axis=3)(x0_1)
    x0_1 = UpSampling2D(size=(2, 2))(x0_1)
    x0_2 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x0_2 = BatchNormalization(axis=3)(x0_2)
    x0_2 = UpSampling2D(size=(4, 4))(x0_2)
    x0 = add([x0_0, x0_1, x0_2])

    x1_0 = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x1_0 = BatchNormalization(axis=3)(x1_0)
    x1_1 = x[1]
    x1_2 = Conv2D(64, 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x1_2 = BatchNormalization(axis=3)(x1_2)
    x1_2 = UpSampling2D(size=(2, 2))(x1_2)
    x1 = add([x1_0, x1_1, x1_2])

    x2_0 = Conv2D(32, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x2_0 = BatchNormalization(axis=3)(x2_0)
    x2_0 = Activation('relu')(x2_0)
    x2_0 = Conv2D(224, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x2_0)
    x2_0 = BatchNormalization(axis=3)(x2_0)
    x2_1 = Conv2D(224, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x2_1 = BatchNormalization(axis=3)(x2_1)
    x2_2 = x[2]
    x2 = add([x2_0, x2_1, x2_2])
    return [x0, x1, x2]


def transition_layer3(x, out_filters_list=[32, 64, 128, 224]):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x0 = BatchNormalization(axis=3)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(out_filters_list[2], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
    x2 = BatchNormalization(axis=3)(x2)
    x2 = Activation('relu')(x2)

    x3 = Conv2D(out_filters_list[3], 3, strides=(2, 2),
                padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
    x3 = BatchNormalization(axis=3)(x3)
    x3 = Activation('relu')(x3)

    return [x0, x1, x2, x3]


def make_branch3_0(x, out_filters=32):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch3_1(x, out_filters=64):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch3_2(x, out_filters=128):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch3_3(x, out_filters=224):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x

def fuse_layer3(x):
    x0_0 = x[0]
    x0_1 = Conv2D(16, 1, use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = UpSampling2D(size=(2, 2))(x0_1)
    x0_2 = Conv2D(16, 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x0_2 = UpSampling2D(size=(4, 4))(x0_2)
    x0_3 = Conv2D(16, 1, use_bias=False, kernel_initializer='he_normal')(x[3])
    x0_3 = UpSampling2D(size=(8, 8))(x0_3)
    x0 = concatenate([x0_0, x0_1, x0_2, x0_3], axis=-1)
    return x0

def final_layer(x, classes=7):
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(classes, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.GlobalAvgPool2D()(x)
    x = ClusteringAffinity(7, 1, 100)(x)
    return x

def seg_hrnet(batch_size=16, input_height=224, input_width=224, channel=3, classes=6):
    inputs = Input(batch_shape=(batch_size,) + (input_height, input_width, channel))

    x = stem_net(inputs)

    x = transition_layer1(x)
    x0 = make_branch1_0(x[0])
    x1 = make_branch1_1(x[1])
    x = fuse_layer1([x0, x1])

    x = transition_layer2(x)
    x0 = make_branch2_0(x[0])
    x1 = make_branch2_1(x[1])
    x2 = make_branch2_2(x[2])
    x = fuse_layer2([x0, x1, x2])

    x = transition_layer3(x)
    x0 = make_branch3_0(x[0])
    x1 = make_branch3_1(x[1])
    x2 = make_branch3_2(x[2])
    x3 = make_branch3_3(x[3])
    x = fuse_layer3([x0, x1, x2, x3])

    x = final_layer(x, classes=classes)
    model = Model(inputs=inputs, outputs=x)
    return inputs, [x0, x1, x2, x3]

