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

def get_kerasxcnnbaseline(nb_classes):
    batch_size = 32
    # nb_classes = 10
    inputYUV = Input(shape=input_shape)
    inputNorm = BatchNormalization(axis=pr_axis)(inputYUV)

    # To simplify the data augmentation, I delay slicing until this point.
    # Not sure if there is a better way to handle it. ---Petar

    inputY = get_slice(pr_axis, 0, input_shape)(inputNorm)
    inputU = get_slice(pr_axis, 1, input_shape)(inputNorm)
    inputV = get_slice(pr_axis, 2, input_shape)(inputNorm)

    convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(inputY)
    convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputU)
    convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputV)

    convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    U_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolU)
    V_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolV)
    Y_to_UV = Convolution2D(32, 1, 1, border_mode='same', activation='relu')(poolY)

    Ymap = merge([poolY, U_to_Y, V_to_Y], mode='concat', concat_axis=pr_axis)
    Umap = merge([poolU, Y_to_UV], mode='concat', concat_axis=pr_axis)
    Vmap = merge([poolV, Y_to_UV], mode='concat', concat_axis=pr_axis)

    convY = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(Ymap)
    convU = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(Umap)
    convV = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(Vmap)

    convY = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convV)

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



    return model, batch_size