from keras.models import Model
from keras.layers import Dense, Dropout, MaxoutDense, Flatten, merge, Input, Lambda
from keras.regularizers import l2
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

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

def get_fitnetxcnn(nb_classes):
    batch_size = 128
    # nb_classes = 10

    inputYUV = Input(shape=input_shape)
    inputNorm = BatchNormalization(axis=pr_axis)(inputYUV)

    inputY = get_slice(pr_axis, 0, input_shape)(inputNorm)
    inputU = get_slice(pr_axis, 1, input_shape)(inputNorm)
    inputV = get_slice(pr_axis, 2, input_shape)(inputNorm)

    inputY_drop = Dropout(0.2)(inputY)
    inputU_drop = Dropout(0.2)(inputU)
    inputV_drop = Dropout(0.2)(inputV)

    h0_conv_Y_a = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputY_drop)
    h0_conv_Y_b = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputY_drop)
    h0_conv_Y = merge([h0_conv_Y_a, h0_conv_Y_b], mode='max', concat_axis=pr_axis)
    h0_conv_Y = BatchNormalization(axis=pr_axis)(h0_conv_Y)

    h0_conv_U_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputU_drop)
    h0_conv_U_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputU_drop)
    h0_conv_U = merge([h0_conv_U_a, h0_conv_U_b], mode='max', concat_axis=pr_axis)
    h0_conv_U = BatchNormalization(axis=pr_axis)(h0_conv_U)

    h0_conv_V_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputV_drop)
    h0_conv_V_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputV_drop)
    h0_conv_V = merge([h0_conv_V_a, h0_conv_V_b], mode='max', concat_axis=pr_axis)
    h0_conv_V = BatchNormalization(axis=pr_axis)(h0_conv_V)

    h1_conv_Y_a = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_Y)
    h1_conv_Y_b = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_Y)
    h1_conv_Y = merge([h1_conv_Y_a, h1_conv_Y_b], mode='max', concat_axis=pr_axis)
    h1_conv_Y = BatchNormalization(axis=pr_axis)(h1_conv_Y)

    h1_conv_U_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_U)
    h1_conv_U_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_U)
    h1_conv_U = merge([h1_conv_U_a, h1_conv_U_b], mode='max', concat_axis=pr_axis)
    h1_conv_U = BatchNormalization(axis=pr_axis)(h1_conv_U)

    h1_conv_V_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_V)
    h1_conv_V_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_V)
    h1_conv_V = merge([h1_conv_V_a, h1_conv_V_b], mode='max', concat_axis=pr_axis)
    h1_conv_V = BatchNormalization(axis=pr_axis)(h1_conv_V)

    h2_conv_Y_a = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_Y)
    h2_conv_Y_b = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_Y)
    h2_conv_Y = merge([h2_conv_Y_a, h2_conv_Y_b], mode='max', concat_axis=pr_axis)
    h2_conv_Y = BatchNormalization(axis=pr_axis)(h2_conv_Y)

    h2_conv_U_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_U)
    h2_conv_U_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_U)
    h2_conv_U = merge([h2_conv_U_a, h2_conv_U_b], mode='max', concat_axis=pr_axis)
    h2_conv_U = BatchNormalization(axis=pr_axis)(h2_conv_U)

    h2_conv_V_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_V)
    h2_conv_V_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_V)
    h2_conv_V = merge([h2_conv_V_a, h2_conv_V_b], mode='max', concat_axis=pr_axis)
    h2_conv_V = BatchNormalization(axis=pr_axis)(h2_conv_V)

    h3_conv_Y_a = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_Y)
    h3_conv_Y_b = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_Y)
    h3_conv_Y = merge([h3_conv_Y_a, h3_conv_Y_b], mode='max', concat_axis=pr_axis)
    h3_conv_Y = BatchNormalization(axis=pr_axis)(h3_conv_Y)

    h3_conv_U_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_U)
    h3_conv_U_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_U)
    h3_conv_U = merge([h3_conv_U_a, h3_conv_U_b], mode='max', concat_axis=pr_axis)
    h3_conv_U = BatchNormalization(axis=pr_axis)(h3_conv_U)

    h3_conv_V_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_V)
    h3_conv_V_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_V)
    h3_conv_V = merge([h3_conv_V_a, h3_conv_V_b], mode='max', concat_axis=pr_axis)
    h3_conv_V = BatchNormalization(axis=pr_axis)(h3_conv_V)

    h4_conv_Y_a = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_Y)
    h4_conv_Y_b = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_Y)
    h4_conv_Y = merge([h4_conv_Y_a, h4_conv_Y_b], mode='max', concat_axis=pr_axis)
    h4_conv_Y = BatchNormalization(axis=pr_axis)(h4_conv_Y)

    h4_conv_U_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_U)
    h4_conv_U_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_U)
    h4_conv_U = merge([h4_conv_U_a, h4_conv_U_b], mode='max', concat_axis=pr_axis)
    h4_conv_U = BatchNormalization(axis=pr_axis)(h4_conv_U)

    h4_conv_V_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_V)
    h4_conv_V_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_V)
    h4_conv_V = merge([h4_conv_V_a, h4_conv_V_b], mode='max', concat_axis=pr_axis)
    h4_conv_V = BatchNormalization(axis=pr_axis)(h4_conv_V)

    poolY = MaxPooling2D(pool_size=(2, 2))(h4_conv_Y)
    poolU = MaxPooling2D(pool_size=(2, 2))(h4_conv_U)
    poolV = MaxPooling2D(pool_size=(2, 2))(h4_conv_V)

    poolY = Dropout(0.2)(poolY)
    poolU = Dropout(0.2)(poolU)
    poolV = Dropout(0.2)(poolV)

    # Inline connections
    Y_to_Y_a = Convolution2D(36, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_Y_b = Convolution2D(36, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_Y = merge([Y_to_Y_a, Y_to_Y_b], mode='max', concat_axis=pr_axis)
    Y_to_Y = BatchNormalization(axis=pr_axis)(Y_to_Y)

    U_to_U_a = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_U_b = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_U = merge([U_to_U_a, U_to_U_b], mode='max', concat_axis=pr_axis)
    U_to_U = BatchNormalization(axis=pr_axis)(U_to_U)

    V_to_V_a = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_V_b = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_V = merge([V_to_V_a, V_to_V_b], mode='max', concat_axis=pr_axis)
    V_to_V = BatchNormalization(axis=pr_axis)(Y_to_Y)

    # Cross connections: Y <-> U, Y <-> V
    Y_to_UV_a = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_UV_b = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_UV = merge([Y_to_UV_a, Y_to_UV_b], mode='max', concat_axis=pr_axis)
    Y_to_UV = BatchNormalization(axis=pr_axis)(Y_to_UV)

    U_to_Y_a = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_Y_b = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_Y = merge([U_to_Y_a, U_to_Y_b], mode='max', concat_axis=pr_axis)
    U_to_Y = BatchNormalization(axis=pr_axis)(U_to_Y)

    V_to_Y_a = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_Y_b = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_Y = merge([V_to_Y_a, V_to_Y_b], mode='max', concat_axis=pr_axis)
    V_to_Y = BatchNormalization(axis=pr_axis)(V_to_Y)

    Ymap = merge([Y_to_Y, U_to_Y, V_to_Y], mode='concat', concat_axis=pr_axis)
    Umap = merge([U_to_U, Y_to_UV], mode='concat', concat_axis=pr_axis)
    Vmap = merge([V_to_V, Y_to_UV], mode='concat', concat_axis=pr_axis)

    h5_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h5_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h5_conv_Y = merge([h5_conv_Y_a, h5_conv_Y_b], mode='max', concat_axis=pr_axis)
    h5_conv_Y = BatchNormalization(axis=pr_axis)(h5_conv_Y)

    h5_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h5_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h5_conv_U = merge([h5_conv_U_a, h5_conv_U_b], mode='max', concat_axis=pr_axis)
    h5_conv_U = BatchNormalization(axis=pr_axis)(h5_conv_U)

    h5_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h5_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h5_conv_V = merge([h5_conv_V_a, h5_conv_V_b], mode='max', concat_axis=pr_axis)
    h5_conv_V = BatchNormalization(axis=pr_axis)(h5_conv_V)

    h6_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_Y)
    h6_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_Y)
    h6_conv_Y = merge([h6_conv_Y_a, h6_conv_Y_b], mode='max', concat_axis=pr_axis)
    h6_conv_Y = BatchNormalization(axis=pr_axis)(h6_conv_Y)

    h6_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_U)
    h6_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_U)
    h6_conv_U = merge([h6_conv_U_a, h6_conv_U_b], mode='max', concat_axis=pr_axis)
    h6_conv_U = BatchNormalization(axis=pr_axis)(h6_conv_U)

    h6_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_V)
    h6_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_V)
    h6_conv_V = merge([h6_conv_V_a, h6_conv_V_b], mode='max', concat_axis=pr_axis)
    h6_conv_V = BatchNormalization(axis=pr_axis)(h6_conv_V)

    h7_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_Y)
    h7_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_Y)
    h7_conv_Y = merge([h7_conv_Y_a, h7_conv_Y_b], mode='max', concat_axis=pr_axis)
    h7_conv_Y = BatchNormalization(axis=pr_axis)(h7_conv_Y)

    h7_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_U)
    h7_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_U)
    h7_conv_U = merge([h7_conv_U_a, h7_conv_U_b], mode='max', concat_axis=pr_axis)
    h7_conv_U = BatchNormalization(axis=pr_axis)(h7_conv_U)

    h7_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_V)
    h7_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_V)
    h7_conv_V = merge([h7_conv_V_a, h7_conv_V_b], mode='max', concat_axis=pr_axis)
    h7_conv_V = BatchNormalization(axis=pr_axis)(h7_conv_V)

    h8_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_Y)
    h8_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_Y)
    h8_conv_Y = merge([h8_conv_Y_a, h8_conv_Y_b], mode='max', concat_axis=pr_axis)
    h8_conv_Y = BatchNormalization(axis=pr_axis)(h8_conv_Y)

    h8_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_U)
    h8_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_U)
    h8_conv_U = merge([h8_conv_U_a, h8_conv_U_b], mode='max', concat_axis=pr_axis)
    h8_conv_U = BatchNormalization(axis=pr_axis)(h8_conv_U)

    h8_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_V)
    h8_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_V)
    h8_conv_V = merge([h8_conv_V_a, h8_conv_V_b], mode='max', concat_axis=pr_axis)
    h8_conv_V = BatchNormalization(axis=pr_axis)(h8_conv_V)

    h9_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_Y)
    h9_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_Y)
    h9_conv_Y = merge([h9_conv_Y_a, h9_conv_Y_b], mode='max', concat_axis=pr_axis)
    h9_conv_Y = BatchNormalization(axis=pr_axis)(h9_conv_Y)

    h9_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_U)
    h9_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_U)
    h9_conv_U = merge([h9_conv_U_a, h9_conv_U_b], mode='max', concat_axis=pr_axis)
    h9_conv_U = BatchNormalization(axis=pr_axis)(h9_conv_U)

    h9_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_V)
    h9_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_V)
    h9_conv_V = merge([h9_conv_V_a, h9_conv_V_b], mode='max', concat_axis=pr_axis)
    h9_conv_V = BatchNormalization(axis=pr_axis)(h9_conv_V)

    h10_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_Y)
    h10_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_Y)
    h10_conv_Y = merge([h10_conv_Y_a, h10_conv_Y_b], mode='max', concat_axis=pr_axis)
    h10_conv_Y = BatchNormalization(axis=pr_axis)(h10_conv_Y)

    h10_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_U)
    h10_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_U)
    h10_conv_U = merge([h10_conv_U_a, h10_conv_U_b], mode='max', concat_axis=pr_axis)
    h10_conv_U = BatchNormalization(axis=pr_axis)(h10_conv_U)

    h10_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_V)
    h10_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_V)
    h10_conv_V = merge([h10_conv_V_a, h10_conv_V_b], mode='max', concat_axis=pr_axis)
    h10_conv_V = BatchNormalization(axis=pr_axis)(h10_conv_V)

    poolY = MaxPooling2D(pool_size=(2, 2))(h10_conv_Y)
    poolU = MaxPooling2D(pool_size=(2, 2))(h10_conv_U)
    poolV = MaxPooling2D(pool_size=(2, 2))(h10_conv_V)

    poolY = Dropout(0.2)(poolY)
    poolU = Dropout(0.2)(poolU)
    poolV = Dropout(0.2)(poolV)

    # Inline connections
    Y_to_Y_a = Convolution2D(60, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_Y_b = Convolution2D(60, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_Y = merge([Y_to_Y_a, Y_to_Y_b], mode='max', concat_axis=pr_axis)
    Y_to_Y = BatchNormalization(axis=pr_axis)(Y_to_Y)

    U_to_U_a = Convolution2D(30, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_U_b = Convolution2D(30, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_U = merge([U_to_U_a, U_to_U_b], mode='max', concat_axis=pr_axis)
    U_to_U = BatchNormalization(axis=pr_axis)(U_to_U)

    V_to_V_a = Convolution2D(30, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_V_b = Convolution2D(30, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_V = merge([V_to_V_a, V_to_V_b], mode='max', concat_axis=pr_axis)
    V_to_V = BatchNormalization(axis=pr_axis)(Y_to_Y)

    # Cross connections: Y <-> U, Y <-> V
    Y_to_UV_a = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_UV_b = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_UV = merge([Y_to_UV_a, Y_to_UV_b], mode='max', concat_axis=pr_axis)
    Y_to_UV = BatchNormalization(axis=pr_axis)(Y_to_UV)

    U_to_Y_a = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_Y_b = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_Y = merge([U_to_Y_a, U_to_Y_b], mode='max', concat_axis=pr_axis)
    U_to_Y = BatchNormalization(axis=pr_axis)(U_to_Y)

    V_to_Y_a = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_Y_b = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_Y = merge([V_to_Y_a, V_to_Y_b], mode='max', concat_axis=pr_axis)
    V_to_Y = BatchNormalization(axis=pr_axis)(V_to_Y)

    Ymap = merge([Y_to_Y, U_to_Y, V_to_Y], mode='concat', concat_axis=pr_axis)
    Umap = merge([U_to_U, Y_to_UV], mode='concat', concat_axis=pr_axis)
    Vmap = merge([V_to_V, Y_to_UV], mode='concat', concat_axis=pr_axis)

    h11_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h11_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h11_conv_Y = merge([h11_conv_Y_a, h11_conv_Y_b], mode='max', concat_axis=pr_axis)
    h11_conv_Y = BatchNormalization(axis=pr_axis)(h11_conv_Y)

    h11_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h11_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h11_conv_U = merge([h11_conv_U_a, h11_conv_U_b], mode='max', concat_axis=pr_axis)
    h11_conv_U = BatchNormalization(axis=pr_axis)(h11_conv_U)

    h11_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h11_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h11_conv_V = merge([h11_conv_V_a, h11_conv_V_b], mode='max', concat_axis=pr_axis)
    h11_conv_V = BatchNormalization(axis=pr_axis)(h11_conv_V)

    h12_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_Y)
    h12_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_Y)
    h12_conv_Y = merge([h12_conv_Y_a, h12_conv_Y_b], mode='max', concat_axis=pr_axis)
    h12_conv_Y = BatchNormalization(axis=pr_axis)(h12_conv_Y)

    h12_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_U)
    h12_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_U)
    h12_conv_U = merge([h12_conv_U_a, h12_conv_U_b], mode='max', concat_axis=pr_axis)
    h12_conv_U = BatchNormalization(axis=pr_axis)(h12_conv_U)

    h12_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_V)
    h12_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_V)
    h12_conv_V = merge([h12_conv_V_a, h12_conv_V_b], mode='max', concat_axis=pr_axis)
    h12_conv_V = BatchNormalization(axis=pr_axis)(h12_conv_V)

    h13_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_Y)
    h13_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_Y)
    h13_conv_Y = merge([h13_conv_Y_a, h13_conv_Y_b], mode='max', concat_axis=pr_axis)
    h13_conv_Y = BatchNormalization(axis=pr_axis)(h13_conv_Y)

    h13_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_U)
    h13_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_U)
    h13_conv_U = merge([h13_conv_U_a, h13_conv_U_b], mode='max', concat_axis=pr_axis)
    h13_conv_U = BatchNormalization(axis=pr_axis)(h13_conv_U)

    h13_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_V)
    h13_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_V)
    h13_conv_V = merge([h13_conv_V_a, h13_conv_V_b], mode='max', concat_axis=pr_axis)
    h13_conv_V = BatchNormalization(axis=pr_axis)(h13_conv_V)

    h14_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_Y)
    h14_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_Y)
    h14_conv_Y = merge([h14_conv_Y_a, h14_conv_Y_b], mode='max', concat_axis=pr_axis)
    h14_conv_Y = BatchNormalization(axis=pr_axis)(h14_conv_Y)

    h14_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_U)
    h14_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_U)
    h14_conv_U = merge([h14_conv_U_a, h14_conv_U_b], mode='max', concat_axis=pr_axis)
    h14_conv_U = BatchNormalization(axis=pr_axis)(h14_conv_U)

    h14_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_V)
    h14_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_V)
    h14_conv_V = merge([h14_conv_V_a, h14_conv_V_b], mode='max', concat_axis=pr_axis)
    h14_conv_V = BatchNormalization(axis=pr_axis)(h14_conv_V)

    h15_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_Y)
    h15_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_Y)
    h15_conv_Y = merge([h15_conv_Y_a, h15_conv_Y_b], mode='max', concat_axis=pr_axis)
    h15_conv_Y = BatchNormalization(axis=pr_axis)(h15_conv_Y)

    h15_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_U)
    h15_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_U)
    h15_conv_U = merge([h15_conv_U_a, h15_conv_U_b], mode='max', concat_axis=pr_axis)
    h15_conv_U = BatchNormalization(axis=pr_axis)(h15_conv_U)

    h15_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_V)
    h15_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_V)
    h15_conv_V = merge([h15_conv_V_a, h15_conv_V_b], mode='max', concat_axis=pr_axis)
    h15_conv_V = BatchNormalization(axis=pr_axis)(h15_conv_V)

    h16_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_Y)
    h16_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_Y)
    h16_conv_Y = merge([h16_conv_Y_a, h16_conv_Y_b], mode='max', concat_axis=pr_axis)
    h16_conv_Y = BatchNormalization(axis=pr_axis)(h16_conv_Y)

    h16_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_U)
    h16_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_U)
    h16_conv_U = merge([h16_conv_U_a, h16_conv_U_b], mode='max', concat_axis=pr_axis)
    h16_conv_U = BatchNormalization(axis=pr_axis)(h16_conv_U)

    h16_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_V)
    h16_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_V)
    h16_conv_V = merge([h16_conv_V_a, h16_conv_V_b], mode='max', concat_axis=pr_axis)
    h16_conv_V = BatchNormalization(axis=pr_axis)(h16_conv_V)

    poolY = MaxPooling2D(pool_size=(8, 8))(h16_conv_Y)
    poolU = MaxPooling2D(pool_size=(8, 8))(h16_conv_U)
    poolV = MaxPooling2D(pool_size=(8, 8))(h16_conv_V)

    poolY = Dropout(0.2)(poolY)
    poolU = Dropout(0.2)(poolU)
    poolV = Dropout(0.2)(poolV)

    concat_map = merge([poolY, poolU, poolV], mode='concat', concat_axis=pr_axis)

    h16 = Flatten()(concat_map)
    h17 = MaxoutDense(500, nb_feature=5, init='glorot_uniform', W_regularizer=l2(0.0005))(h16)
    h17 = BatchNormalization(axis=1)(h17)
    h17_drop = Dropout(0.2)(h17)
    out = Dense(nb_classes, activation='softmax', init='glorot_uniform', W_regularizer=l2(0.0005))(h17_drop)

    model = Model(input=inputYUV, output=out)

    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])

    return model, batch_size
