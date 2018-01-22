from keras.models import Model
from keras.layers import Input, Lambda, BatchNormalization, Convolution2D, MaxPooling2D, Dropout, merge, Flatten, Dense

import keras.backend as K
input_shape = (3, 32, 32)
pr_axis = 1
if K.image_dim_ordering() == 'tf':
    input_shape = (32, 32, 3)
    pr_axis = 3

def get_slice(axis, axis_id, input_shape):
    return Lambda(
        lambda x: x[[slice(None) if i != axis else slice(axis_id, axis_id + 1) for i in range(len(input_shape) + 1)]],
        output_shape=[p if i+1 != axis else 1 for i, p in enumerate(input_shape)])


def get_eqsplit_eq_xcon(nb_classes):
    inputYUV = Input(shape=input_shape)
    # inputNorm = BatchNormalization(axis=1)(inputYUV)

    inputY = get_slice(pr_axis, 0, input_shape)(inputYUV)
    inputU = get_slice(pr_axis, 1, input_shape)(inputYUV)
    inputV = get_slice(pr_axis, 2, input_shape)(inputYUV)

    inputYnorm = BatchNormalization(axis=pr_axis)(inputY)
    inputUnorm = BatchNormalization(axis=pr_axis)(inputU)
    inputVnorm = BatchNormalization(axis=pr_axis)(inputV)

    convY = Convolution2D(21, 3, 3, border_mode='same', activation='relu')(inputYnorm)
    convU = Convolution2D(21, 3, 3, border_mode='same', activation='relu')(inputUnorm)
    convV = Convolution2D(21, 3, 3, border_mode='same', activation='relu')(inputVnorm)

    convY = Convolution2D(21, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(21, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(21, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    # ------------------

    Y_to_U = Convolution2D(21, 1, 1, border_mode='same', activation='relu')(poolY)
    Y_to_V = Convolution2D(21, 1, 1, border_mode='same', activation='relu')(poolY)

    U_to_Y = Convolution2D(21, 1, 1, border_mode='same', activation='relu')(poolU)
    U_to_V = Convolution2D(21, 1, 1, border_mode='same', activation='relu')(poolU)

    V_to_Y = Convolution2D(21, 1, 1, border_mode='same', activation='relu')(poolV)
    V_to_U = Convolution2D(21, 1, 1, border_mode='same', activation='relu')(poolV)

    Ymap = merge([poolY, U_to_Y, V_to_Y], mode='concat', concat_axis=pr_axis)
    Umap = merge([poolU, V_to_U, Y_to_U], mode='concat', concat_axis=pr_axis)
    Vmap = merge([poolV, Y_to_V, U_to_V], mode='concat', concat_axis=pr_axis)

    Ycon = Ymap
    Ucon = Umap
    Vcon = Vmap

    # ------------------

    convY = Convolution2D(42, 3, 3, border_mode='same', activation='relu')(Ycon)
    convU = Convolution2D(42, 3, 3, border_mode='same', activation='relu')(Ucon)
    convV = Convolution2D(42, 3, 3, border_mode='same', activation='relu')(Vcon)

    convY = Convolution2D(42, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(42, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(42, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    concatenate_map = merge([poolY, poolU, poolV], mode='concat', concat_axis=pr_axis)

    reshape = Flatten()(concatenate_map)
    fc = Dense(512, activation='relu')(reshape)
    fc = Dropout(0.5)(fc)
    out = Dense(nb_classes, activation='softmax')(fc)

    model = Model(input=inputYUV, output=out)
    return model, 32
