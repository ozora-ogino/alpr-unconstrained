from tensorflow.keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Input, MaxPooling2D
from tensorflow.keras.models import Model

from darkflow.net.build import TFNet


def res_block(x, sz, filter_sz=3, in_conv_size=1):
    xi = x
    for _ in range(in_conv_size):
        xi = Conv2D(sz, filter_sz, activation="linear", padding="same")(xi)
        xi = BatchNormalization()(xi)
        xi = Activation("relu")(xi)
    xi = Conv2D(sz, filter_sz, activation="linear", padding="same")(xi)
    xi = BatchNormalization()(xi)
    xi = Add()([xi, x])
    xi = Activation("relu")(xi)
    return xi


def conv_batch(_input, fsz, csz, activation="relu", padding="same", strides=(1, 1)):
    output = Conv2D(fsz, csz, activation="linear", padding=padding, strides=strides)(_input)
    output = BatchNormalization()(output)
    output = Activation(activation)(output)
    return output


def end_block(x):
    xprobs = Conv2D(2, 3, activation="softmax", padding="same")(x)
    xbbox = Conv2D(6, 3, activation="linear", padding="same")(x)
    return Concatenate(3)([xprobs, xbbox])


def build_model():

    input_layer = Input(shape=(None, None, 3), name="input")

    x = conv_batch(input_layer, 16, 3)
    x = conv_batch(x, 16, 3)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = conv_batch(x, 32, 3)
    x = res_block(x, 32)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = conv_batch(x, 64, 3)
    x = res_block(x, 64)
    x = res_block(x, 64)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = conv_batch(x, 64, 3)
    x = res_block(x, 64)
    x = res_block(x, 64)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = conv_batch(x, 128, 3)
    x = res_block(x, 128)
    x = res_block(x, 128)
    x = res_block(x, 128)
    x = res_block(x, 128)

    x = end_block(x)

    return Model(inputs=input_layer, outputs=x)


def load_yolov2(model_cfg: str = None, model_weights: str = None):
    options = {
        "model": model_cfg or "/opt/darkflow/cfg/yolov2.cfg",
        "load": model_weights or "/opt/bin/yolov2.weights",
        "threshold": 0.1,
    }
    detect = TFNet(options)
    return detect
