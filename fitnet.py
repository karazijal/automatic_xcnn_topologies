from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout, Flatten, Lambda, Input, merge, MaxoutDense
from keras.regularizers import l2
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

import keras.backend as K

def get_fitnet(nb_classes, input_shape, ch_axis):
    inputYUV = Input(shape=input_shape)
    inputNorm = BatchNormalization(axis=ch_axis)(inputYUV)

    # input_drop = Dropout(0.2)(inputNorm)

    # This is a single convolutional maxout layer.
    h0_conv_a = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputNorm)
    h0_conv_b = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputNorm)
    h0_conv = merge([h0_conv_a, h0_conv_b], mode='max', concat_axis=ch_axis)
    h0_conv = BatchNormalization(axis=ch_axis)(h0_conv)

    h1_conv_a = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h0_conv)
    h1_conv_b = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h0_conv)
    h1_conv = merge([h1_conv_a, h1_conv_b], mode='max', concat_axis=ch_axis)
    h1_conv = BatchNormalization(axis=ch_axis)(h1_conv)

    h2_conv_a = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h1_conv)
    h2_conv_b = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h1_conv)
    h2_conv = merge([h2_conv_a, h2_conv_b], mode='max', concat_axis=ch_axis)
    h2_conv = BatchNormalization(axis=ch_axis)(h2_conv)

    h3_conv_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h2_conv)
    h3_conv_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h2_conv)
    h3_conv = merge([h3_conv_a, h3_conv_b], mode='max', concat_axis=ch_axis)
    h3_conv = BatchNormalization(axis=ch_axis)(h3_conv)

    h4_conv_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h3_conv)
    h4_conv_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h3_conv)
    h4_conv = merge([h4_conv_a, h4_conv_b], mode='max', concat_axis=ch_axis)
    h4_conv = BatchNormalization(axis=ch_axis)(h4_conv)

    h4_pool = MaxPooling2D(pool_size=(2, 2), name='poolspot_1')(h4_conv)
    h4_drop = Dropout(0.2)(h4_pool)

    h5_conv_a = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h4_drop)
    h5_conv_b = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h4_drop)
    h5_conv = merge([h5_conv_a, h5_conv_b], mode='max', concat_axis=ch_axis)
    h5_conv = BatchNormalization(axis=ch_axis)(h5_conv)

    h6_conv_a = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h5_conv)
    h6_conv_b = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h5_conv)
    h6_conv = merge([h6_conv_a, h6_conv_b], mode='max', concat_axis=ch_axis)
    h6_conv = BatchNormalization(axis=ch_axis)(h6_conv)

    h7_conv_a = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h6_conv)
    h7_conv_b = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h6_conv)
    h7_conv = merge([h7_conv_a, h7_conv_b], mode='max', concat_axis=ch_axis)
    h7_conv = BatchNormalization(axis=ch_axis)(h7_conv)

    h8_conv_a = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h7_conv)
    h8_conv_b = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h7_conv)
    h8_conv = merge([h8_conv_a, h8_conv_b], mode='max', concat_axis=ch_axis)
    h8_conv = BatchNormalization(axis=ch_axis)(h8_conv)

    h9_conv_a = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h8_conv)
    h9_conv_b = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h8_conv)
    h9_conv = merge([h9_conv_a, h9_conv_b], mode='max', concat_axis=ch_axis)
    h9_conv = BatchNormalization(axis=ch_axis)(h9_conv)

    h10_conv_a = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h9_conv)
    h10_conv_b = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h9_conv)
    h10_conv = merge([h10_conv_a, h10_conv_b], mode='max', concat_axis=ch_axis)
    h10_conv = BatchNormalization(axis=ch_axis)(h10_conv)

    h10_pool = MaxPooling2D(pool_size=(2, 2), name='poolspot_2')(h10_conv)
    h10_drop = Dropout(0.2)(h10_pool)

    h11_conv_a = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h10_drop)
    h11_conv_b = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h10_drop)
    h11_conv = merge([h11_conv_a, h11_conv_b], mode='max', concat_axis=ch_axis)
    h11_conv = BatchNormalization(axis=ch_axis)(h11_conv)

    h12_conv_a = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h11_conv)
    h12_conv_b = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h11_conv)
    h12_conv = merge([h12_conv_a, h12_conv_b], mode='max', concat_axis=ch_axis)
    h12_conv = BatchNormalization(axis=ch_axis)(h12_conv)

    h13_conv_a = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h12_conv)
    h13_conv_b = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h12_conv)
    h13_conv = merge([h13_conv_a, h13_conv_b], mode='max', concat_axis=ch_axis)
    h13_conv = BatchNormalization(axis=ch_axis)(h13_conv)

    h14_conv_a = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h13_conv)
    h14_conv_b = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h13_conv)
    h14_conv = merge([h14_conv_a, h14_conv_b], mode='max', concat_axis=ch_axis)
    h14_conv = BatchNormalization(axis=ch_axis)(h14_conv)

    h15_conv_a = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h14_conv)
    h15_conv_b = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h14_conv)
    h15_conv = merge([h15_conv_a, h15_conv_b], mode='max', concat_axis=ch_axis)
    h15_conv = BatchNormalization(axis=ch_axis)(h15_conv)

    h16_conv_a = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h15_conv)
    h16_conv_b = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h15_conv)
    h16_conv = merge([h16_conv_a, h16_conv_b], mode='max', concat_axis=ch_axis)
    h16_conv = BatchNormalization(axis=ch_axis)(h16_conv)

    h16_pool = MaxPooling2D(pool_size=(8, 8))(h16_conv)
    h16_drop = Dropout(0.2)(h16_pool)
    # h165 = merge()
    h16 = Flatten()(h16_drop)
    h17 = MaxoutDense(500, nb_feature=5, init='glorot_uniform', W_regularizer=l2(0.0005))(h16)
    h17 = BatchNormalization(axis=1)(h17)
    # h17_drop = Dropout(0.2)(h17)
    out = Dense(nb_classes, activation='softmax', init='glorot_uniform', W_regularizer=l2(0.0005))(h17)

    model = Model(input=inputYUV, output=out)
    return model
    
    
def get_xfitnet_full(nb_classes, input_shape, ch_axis):
    inputYUV = Input(shape=input_shape)
    inputNorm = BatchNormalization(axis=ch_axis)(inputYUV)

    inp = inputNorm
    if K.image_dim_ordering() == 'tf':
        lb1 = Lambda(lambda x: x[:, :, :, 0:1], output_shape=(32, 32, 1))(inp)
        lb2 = Lambda(lambda x: x[:, :, :, 1:2], output_shape=(32, 32, 1))(inp)
        lb3 = Lambda(lambda x: x[:, :, :, 2:3], output_shape=(32, 32, 1))(inp)
    else:
        lb1 = Lambda(lambda x: x[:, 0:1, :, :], output_shape=(1, 32, 32))(inp)
        lb2 = Lambda(lambda x: x[:, 1:2, :, :], output_shape=(1, 32, 32))(inp)
        lb3 = Lambda(lambda x: x[:, 2:3, :, :], output_shape=(1, 32, 32))(inp)

    
    inputY = lb1
    inputU = lb2
    inputV = lb3
    
    inputY_drop = Dropout(0.2)(inputY)
    inputU_drop = Dropout(0.2)(inputU)
    inputV_drop = Dropout(0.2)(inputV)
    
    h0_conv_Y_a = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputY)
    h0_conv_Y_b = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputY)
    h0_conv_Y = merge([h0_conv_Y_a, h0_conv_Y_b], mode='max', concat_axis=ch_axis)
    h0_conv_Y = BatchNormalization(axis=ch_axis)(h0_conv_Y)
    
    h0_conv_U_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputU)
    h0_conv_U_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputU)
    h0_conv_U = merge([h0_conv_U_a, h0_conv_U_b], mode='max', concat_axis=ch_axis)
    h0_conv_U = BatchNormalization(axis=ch_axis)(h0_conv_U)
    
    h0_conv_V_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputV)
    h0_conv_V_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputV)
    h0_conv_V = merge([h0_conv_V_a, h0_conv_V_b], mode='max', concat_axis=ch_axis)
    h0_conv_V = BatchNormalization(axis=ch_axis)(h0_conv_V)
    
    h1_conv_Y_a = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h0_conv_Y)
    h1_conv_Y_b = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h0_conv_Y)
    h1_conv_Y = merge([h1_conv_Y_a, h1_conv_Y_b], mode='max', concat_axis=ch_axis)
    h1_conv_Y = BatchNormalization(axis=ch_axis)(h1_conv_Y)
    
    h1_conv_U_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h0_conv_U)
    h1_conv_U_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h0_conv_U)
    h1_conv_U = merge([h1_conv_U_a, h1_conv_U_b], mode='max', concat_axis=ch_axis)
    h1_conv_U = BatchNormalization(axis=ch_axis)(h1_conv_U)
    
    h1_conv_V_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h0_conv_V)
    h1_conv_V_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h0_conv_V)
    h1_conv_V = merge([h1_conv_V_a, h1_conv_V_b], mode='max', concat_axis=ch_axis)
    h1_conv_V = BatchNormalization(axis=ch_axis)(h1_conv_V)
    
    h2_conv_Y_a = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h1_conv_Y)
    h2_conv_Y_b = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h1_conv_Y)
    h2_conv_Y = merge([h2_conv_Y_a, h2_conv_Y_b], mode='max', concat_axis=ch_axis)
    h2_conv_Y = BatchNormalization(axis=ch_axis)(h2_conv_Y)
    
    h2_conv_U_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h1_conv_U)
    h2_conv_U_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h1_conv_U)
    h2_conv_U = merge([h2_conv_U_a, h2_conv_U_b], mode='max', concat_axis=ch_axis)
    h2_conv_U = BatchNormalization(axis=ch_axis)(h2_conv_U)
    
    h2_conv_V_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h1_conv_V)
    h2_conv_V_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h1_conv_V)
    h2_conv_V = merge([h2_conv_V_a, h2_conv_V_b], mode='max', concat_axis=ch_axis)
    h2_conv_V = BatchNormalization(axis=ch_axis)(h2_conv_V)
    
    h3_conv_Y_a = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h2_conv_Y)
    h3_conv_Y_b = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h2_conv_Y)
    h3_conv_Y = merge([h3_conv_Y_a, h3_conv_Y_b], mode='max', concat_axis=ch_axis)
    h3_conv_Y = BatchNormalization(axis=ch_axis)(h3_conv_Y)
    
    h3_conv_U_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h2_conv_U)
    h3_conv_U_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h2_conv_U)
    h3_conv_U = merge([h3_conv_U_a, h3_conv_U_b], mode='max', concat_axis=ch_axis)
    h3_conv_U = BatchNormalization(axis=ch_axis)(h3_conv_U)
    
    h3_conv_V_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h2_conv_V)
    h3_conv_V_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h2_conv_V)
    h3_conv_V = merge([h3_conv_V_a, h3_conv_V_b], mode='max', concat_axis=ch_axis)
    h3_conv_V = BatchNormalization(axis=ch_axis)(h3_conv_V)
    
    h4_conv_Y_a = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h3_conv_Y)
    h4_conv_Y_b = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h3_conv_Y)
    h4_conv_Y = merge([h4_conv_Y_a, h4_conv_Y_b], mode='max', concat_axis=ch_axis)
    h4_conv_Y = BatchNormalization(axis=ch_axis)(h4_conv_Y)
    
    h4_conv_U_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h3_conv_U)
    h4_conv_U_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h3_conv_U)
    h4_conv_U = merge([h4_conv_U_a, h4_conv_U_b], mode='max', concat_axis=ch_axis)
    h4_conv_U = BatchNormalization(axis=ch_axis)(h4_conv_U)
    
    h4_conv_V_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h3_conv_V)
    h4_conv_V_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h3_conv_V)
    h4_conv_V = merge([h4_conv_V_a, h4_conv_V_b], mode='max', concat_axis=ch_axis)
    h4_conv_V = BatchNormalization(axis=ch_axis)(h4_conv_V)
    
    poolY = MaxPooling2D(pool_size=(2, 2))(h4_conv_Y)
    poolU = MaxPooling2D(pool_size=(2, 2))(h4_conv_U)
    poolV = MaxPooling2D(pool_size=(2, 2))(h4_conv_V)
    
    poolY = Dropout(0.2)(poolY)
    poolU = Dropout(0.2)(poolU)
    poolV = Dropout(0.2)(poolV)
    
    # Inline connections
    Y_to_Y_a = Convolution2D(36, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_Y_b = Convolution2D(36, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_Y = merge([Y_to_Y_a, Y_to_Y_b], mode='max', concat_axis=ch_axis)
    Y_to_Y = BatchNormalization(axis=ch_axis)(Y_to_Y)
    
    U_to_U_a = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_U_b = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_U = merge([U_to_U_a, U_to_U_b], mode='max', concat_axis=ch_axis)
    U_to_U = BatchNormalization(axis=ch_axis)(U_to_U)
    
    V_to_V_a = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_V_b = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_V = merge([V_to_V_a, V_to_V_b], mode='max', concat_axis=ch_axis)
    V_to_V = BatchNormalization(axis=ch_axis)(Y_to_Y)
    
    # Cross connections: Y <-> U, Y <-> V
    Y_to_UV_a = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_UV_b = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_UV = merge([Y_to_UV_a, Y_to_UV_b], mode='max', concat_axis=ch_axis)
    Y_to_UV = BatchNormalization(axis=ch_axis)(Y_to_UV)
    
    U_to_Y_a = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_Y_b = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_Y = merge([U_to_Y_a, U_to_Y_b], mode='max', concat_axis=ch_axis)
    U_to_Y = BatchNormalization(axis=ch_axis)(U_to_Y)
    
    V_to_Y_a = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_Y_b = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_Y = merge([V_to_Y_a, V_to_Y_b], mode='max', concat_axis=ch_axis)
    V_to_Y = BatchNormalization(axis=ch_axis)(V_to_Y)
    
    Ymap = merge([Y_to_Y, U_to_Y, V_to_Y], mode='concat', concat_axis=ch_axis)
    Umap = merge([U_to_U, Y_to_UV], mode='concat', concat_axis=ch_axis)
    Vmap = merge([V_to_V, Y_to_UV], mode='concat', concat_axis=ch_axis)
    
    h5_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h5_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h5_conv_Y = merge([h5_conv_Y_a, h5_conv_Y_b], mode='max', concat_axis=ch_axis)
    h5_conv_Y = BatchNormalization(axis=ch_axis)(h5_conv_Y)
    
    h5_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h5_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h5_conv_U = merge([h5_conv_U_a, h5_conv_U_b], mode='max', concat_axis=ch_axis)
    h5_conv_U = BatchNormalization(axis=ch_axis)(h5_conv_U)
    
    h5_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h5_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h5_conv_V = merge([h5_conv_V_a, h5_conv_V_b], mode='max', concat_axis=ch_axis)
    h5_conv_V = BatchNormalization(axis=ch_axis)(h5_conv_V)
    
    h6_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h5_conv_Y)
    h6_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h5_conv_Y)
    h6_conv_Y = merge([h6_conv_Y_a, h6_conv_Y_b], mode='max', concat_axis=ch_axis)
    h6_conv_Y = BatchNormalization(axis=ch_axis)(h6_conv_Y)
    
    h6_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h5_conv_U)
    h6_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h5_conv_U)
    h6_conv_U = merge([h6_conv_U_a, h6_conv_U_b], mode='max', concat_axis=ch_axis)
    h6_conv_U = BatchNormalization(axis=ch_axis)(h6_conv_U)
    
    h6_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h5_conv_V)
    h6_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h5_conv_V)
    h6_conv_V = merge([h6_conv_V_a, h6_conv_V_b], mode='max', concat_axis=ch_axis)
    h6_conv_V = BatchNormalization(axis=ch_axis)(h6_conv_V)
    
    h7_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h6_conv_Y)
    h7_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h6_conv_Y)
    h7_conv_Y = merge([h7_conv_Y_a, h7_conv_Y_b], mode='max', concat_axis=ch_axis)
    h7_conv_Y = BatchNormalization(axis=ch_axis)(h7_conv_Y)
    
    h7_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h6_conv_U)
    h7_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h6_conv_U)
    h7_conv_U = merge([h7_conv_U_a, h7_conv_U_b], mode='max', concat_axis=ch_axis)
    h7_conv_U = BatchNormalization(axis=ch_axis)(h7_conv_U)
    
    h7_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h6_conv_V)
    h7_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h6_conv_V)
    h7_conv_V = merge([h7_conv_V_a, h7_conv_V_b], mode='max', concat_axis=ch_axis)
    h7_conv_V = BatchNormalization(axis=ch_axis)(h7_conv_V)
    
    h8_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h7_conv_Y)
    h8_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h7_conv_Y)
    h8_conv_Y = merge([h8_conv_Y_a, h8_conv_Y_b], mode='max', concat_axis=ch_axis)
    h8_conv_Y = BatchNormalization(axis=ch_axis)(h8_conv_Y)
    
    h8_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h7_conv_U)
    h8_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h7_conv_U)
    h8_conv_U = merge([h8_conv_U_a, h8_conv_U_b], mode='max', concat_axis=ch_axis)
    h8_conv_U = BatchNormalization(axis=ch_axis)(h8_conv_U)
    
    h8_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h7_conv_V)
    h8_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h7_conv_V)
    h8_conv_V = merge([h8_conv_V_a, h8_conv_V_b], mode='max', concat_axis=ch_axis)
    h8_conv_V = BatchNormalization(axis=ch_axis)(h8_conv_V)
    
    h9_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h8_conv_Y)
    h9_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h8_conv_Y)
    h9_conv_Y = merge([h9_conv_Y_a, h9_conv_Y_b], mode='max', concat_axis=ch_axis)
    h9_conv_Y = BatchNormalization(axis=ch_axis)(h9_conv_Y)
    
    h9_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h8_conv_U)
    h9_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h8_conv_U)
    h9_conv_U = merge([h9_conv_U_a, h9_conv_U_b], mode='max', concat_axis=ch_axis)
    h9_conv_U = BatchNormalization(axis=ch_axis)(h9_conv_U)
    
    h9_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h8_conv_V)
    h9_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h8_conv_V)
    h9_conv_V = merge([h9_conv_V_a, h9_conv_V_b], mode='max', concat_axis=ch_axis)
    h9_conv_V = BatchNormalization(axis=ch_axis)(h9_conv_V)
    
    h10_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h9_conv_Y)
    h10_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h9_conv_Y)
    h10_conv_Y = merge([h10_conv_Y_a, h10_conv_Y_b], mode='max', concat_axis=ch_axis)
    h10_conv_Y = BatchNormalization(axis=ch_axis)(h10_conv_Y)
    
    h10_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h9_conv_U)
    h10_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h9_conv_U)
    h10_conv_U = merge([h10_conv_U_a, h10_conv_U_b], mode='max', concat_axis=ch_axis)
    h10_conv_U = BatchNormalization(axis=ch_axis)(h10_conv_U)
    
    h10_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h9_conv_V)
    h10_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h9_conv_V)
    h10_conv_V = merge([h10_conv_V_a, h10_conv_V_b], mode='max', concat_axis=ch_axis)
    h10_conv_V = BatchNormalization(axis=ch_axis)(h10_conv_V)
    
    poolY = MaxPooling2D(pool_size=(2, 2))(h10_conv_Y)
    poolU = MaxPooling2D(pool_size=(2, 2))(h10_conv_U)
    poolV = MaxPooling2D(pool_size=(2, 2))(h10_conv_V)
    
    poolY = Dropout(0.2)(poolY)
    poolU = Dropout(0.2)(poolU)
    poolV = Dropout(0.2)(poolV)

    U_to_Y = Convolution2D(30, 1, 1, border_mode='same', activation='relu')(poolU)
    # U_to_Y = Convolution2D(8, 1, 1, border_mode='same', activation='relu')(U_to_Y)
    U_to_V = Convolution2D(30, 1, 1, border_mode='same', activation='relu')(poolU)
    # U_to_V = Convolution2D(8, 1, 1, border_mode='same', activation='relu')(U_to_V)

    V_to_Y = Convolution2D(30, 1, 1, border_mode='same', activation='relu')(poolV)
    # V_to_Y = Convolution2D(8, 1, 1, border_mode='same', activation='relu')(V_to_Y)
    V_to_U = Convolution2D(30, 1, 1, border_mode='same', activation='relu')(poolV)
    # V_to_U = Convolution2D(8, 1, 1, border_mode='same', activation='relu')(V_to_U)

    Y_to_U = Convolution2D(60, 1, 1, border_mode='same', activation='relu')(poolY)
    # Y_to_U = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(Y_to_U)
    Y_to_V = Convolution2D(60, 1, 1, border_mode='same', activation='relu')(poolY)
    # Y_to_V = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(Y_to_V)

    Ymap = merge([poolY, U_to_Y, V_to_Y], mode='concat', concat_axis=ch_axis)
    Umap = merge([poolU, Y_to_U, V_to_U], mode='concat', concat_axis=ch_axis)
    Vmap = merge([poolV, Y_to_V, U_to_V], mode='concat', concat_axis=ch_axis)
    
    h11_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h11_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h11_conv_Y = merge([h11_conv_Y_a, h11_conv_Y_b], mode='max', concat_axis=ch_axis)
    h11_conv_Y = BatchNormalization(axis=ch_axis)(h11_conv_Y)
    
    h11_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h11_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h11_conv_U = merge([h11_conv_U_a, h11_conv_U_b], mode='max', concat_axis=ch_axis)
    h11_conv_U = BatchNormalization(axis=ch_axis)(h11_conv_U)
    
    h11_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h11_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h11_conv_V = merge([h11_conv_V_a, h11_conv_V_b], mode='max', concat_axis=ch_axis)
    h11_conv_V = BatchNormalization(axis=ch_axis)(h11_conv_V)
    
    h12_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h11_conv_Y)
    h12_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h11_conv_Y)
    h12_conv_Y = merge([h12_conv_Y_a, h12_conv_Y_b], mode='max', concat_axis=ch_axis)
    h12_conv_Y = BatchNormalization(axis=ch_axis)(h12_conv_Y)
    
    h12_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h11_conv_U)
    h12_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h11_conv_U)
    h12_conv_U = merge([h12_conv_U_a, h12_conv_U_b], mode='max', concat_axis=ch_axis)
    h12_conv_U = BatchNormalization(axis=ch_axis)(h12_conv_U)
    
    h12_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h11_conv_V)
    h12_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h11_conv_V)
    h12_conv_V = merge([h12_conv_V_a, h12_conv_V_b], mode='max', concat_axis=ch_axis)
    h12_conv_V = BatchNormalization(axis=ch_axis)(h12_conv_V)
    
    h13_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h12_conv_Y)
    h13_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h12_conv_Y)
    h13_conv_Y = merge([h13_conv_Y_a, h13_conv_Y_b], mode='max', concat_axis=ch_axis)
    h13_conv_Y = BatchNormalization(axis=ch_axis)(h13_conv_Y)
    
    h13_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h12_conv_U)
    h13_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h12_conv_U)
    h13_conv_U = merge([h13_conv_U_a, h13_conv_U_b], mode='max', concat_axis=ch_axis)
    h13_conv_U = BatchNormalization(axis=ch_axis)(h13_conv_U)
    
    h13_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h12_conv_V)
    h13_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h12_conv_V)
    h13_conv_V = merge([h13_conv_V_a, h13_conv_V_b], mode='max', concat_axis=ch_axis)
    h13_conv_V = BatchNormalization(axis=ch_axis)(h13_conv_V)
    
    h14_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h13_conv_Y)
    h14_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h13_conv_Y)
    h14_conv_Y = merge([h14_conv_Y_a, h14_conv_Y_b], mode='max', concat_axis=ch_axis)
    h14_conv_Y = BatchNormalization(axis=ch_axis)(h14_conv_Y)
    
    h14_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h13_conv_U)
    h14_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h13_conv_U)
    h14_conv_U = merge([h14_conv_U_a, h14_conv_U_b], mode='max', concat_axis=ch_axis)
    h14_conv_U = BatchNormalization(axis=ch_axis)(h14_conv_U)
    
    h14_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h13_conv_V)
    h14_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h13_conv_V)
    h14_conv_V = merge([h14_conv_V_a, h14_conv_V_b], mode='max', concat_axis=ch_axis)
    h14_conv_V = BatchNormalization(axis=ch_axis)(h14_conv_V)
    
    h15_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h14_conv_Y)
    h15_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h14_conv_Y)
    h15_conv_Y = merge([h15_conv_Y_a, h15_conv_Y_b], mode='max', concat_axis=ch_axis)
    h15_conv_Y = BatchNormalization(axis=ch_axis)(h15_conv_Y)
    
    h15_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h14_conv_U)
    h15_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h14_conv_U)
    h15_conv_U = merge([h15_conv_U_a, h15_conv_U_b], mode='max', concat_axis=ch_axis)
    h15_conv_U = BatchNormalization(axis=ch_axis)(h15_conv_U)
    
    h15_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h14_conv_V)
    h15_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h14_conv_V)
    h15_conv_V = merge([h15_conv_V_a, h15_conv_V_b], mode='max', concat_axis=ch_axis)
    h15_conv_V = BatchNormalization(axis=ch_axis)(h15_conv_V)
    
    h16_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h15_conv_Y)
    h16_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h15_conv_Y)
    h16_conv_Y = merge([h16_conv_Y_a, h16_conv_Y_b], mode='max', concat_axis=ch_axis)
    h16_conv_Y = BatchNormalization(axis=ch_axis)(h16_conv_Y)
    
    h16_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h15_conv_U)
    h16_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h15_conv_U)
    h16_conv_U = merge([h16_conv_U_a, h16_conv_U_b], mode='max', concat_axis=ch_axis)
    h16_conv_U = BatchNormalization(axis=ch_axis)(h16_conv_U)
    
    h16_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h15_conv_V)
    h16_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h15_conv_V)
    h16_conv_V = merge([h16_conv_V_a, h16_conv_V_b], mode='max', concat_axis=ch_axis)
    h16_conv_V = BatchNormalization(axis=ch_axis)(h16_conv_V)
    
    poolY = MaxPooling2D(pool_size=(8, 8))(h16_conv_Y)
    poolU = MaxPooling2D(pool_size=(8, 8))(h16_conv_U)
    poolV = MaxPooling2D(pool_size=(8, 8))(h16_conv_V)
    
    poolY = Dropout(0.2)(poolY)
    poolU = Dropout(0.2)(poolU)
    poolV = Dropout(0.2)(poolV)
    
    concat_map = merge([poolY, poolU, poolV], mode='concat', concat_axis=ch_axis)
    
    h16 = Flatten()(concat_map)
    h17 = MaxoutDense(500, nb_feature=5, init='glorot_uniform', W_regularizer=l2(0.0005))(h16)
    h17 = BatchNormalization(axis=1)(h17)
    h17_drop = Dropout(0.2)(h17)
    out = Dense(nb_classes, activation='softmax', init='glorot_uniform', W_regularizer=l2(0.0005))(h17)
    
    model = Model(input=inputYUV, output=out)
    return model


def get_xfitnet(nb_classes, input_shape, ch_axis):
    inputYUV = Input(shape=input_shape)
    inputNorm = BatchNormalization(axis=ch_axis)(inputYUV)

    inp = inputNorm
    if K.image_dim_ordering() == 'tf':
        lb1 = Lambda(lambda x: x[:, :, :, 0:1], output_shape=(32, 32, 1))(inp)
        lb2 = Lambda(lambda x: x[:, :, :, 1:2], output_shape=(32, 32, 1))(inp)
        lb3 = Lambda(lambda x: x[:, :, :, 2:3], output_shape=(32, 32, 1))(inp)
    else:
        lb1 = Lambda(lambda x: x[:, 0:1, :, :], output_shape=(1, 32, 32))(inp)
        lb2 = Lambda(lambda x: x[:, 1:2, :, :], output_shape=(1, 32, 32))(inp)
        lb3 = Lambda(lambda x: x[:, 2:3, :, :], output_shape=(1, 32, 32))(inp)

    inputY = lb1
    inputU = lb2
    inputV = lb3

    inputY_drop = Dropout(0.2)(inputY)
    inputU_drop = Dropout(0.2)(inputU)
    inputV_drop = Dropout(0.2)(inputV)

    h0_conv_Y_a = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputY)
    h0_conv_Y_b = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputY)
    h0_conv_Y = merge([h0_conv_Y_a, h0_conv_Y_b], mode='max', concat_axis=ch_axis)
    h0_conv_Y = BatchNormalization(axis=ch_axis)(h0_conv_Y)

    h0_conv_U_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputU)
    h0_conv_U_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputU)
    h0_conv_U = merge([h0_conv_U_a, h0_conv_U_b], mode='max', concat_axis=ch_axis)
    h0_conv_U = BatchNormalization(axis=ch_axis)(h0_conv_U)

    h0_conv_V_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputV)
    h0_conv_V_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputV)
    h0_conv_V = merge([h0_conv_V_a, h0_conv_V_b], mode='max', concat_axis=ch_axis)
    h0_conv_V = BatchNormalization(axis=ch_axis)(h0_conv_V)

    h1_conv_Y_a = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_Y)
    h1_conv_Y_b = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_Y)
    h1_conv_Y = merge([h1_conv_Y_a, h1_conv_Y_b], mode='max', concat_axis=ch_axis)
    h1_conv_Y = BatchNormalization(axis=ch_axis)(h1_conv_Y)

    h1_conv_U_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_U)
    h1_conv_U_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_U)
    h1_conv_U = merge([h1_conv_U_a, h1_conv_U_b], mode='max', concat_axis=ch_axis)
    h1_conv_U = BatchNormalization(axis=ch_axis)(h1_conv_U)

    h1_conv_V_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_V)
    h1_conv_V_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_V)
    h1_conv_V = merge([h1_conv_V_a, h1_conv_V_b], mode='max', concat_axis=ch_axis)
    h1_conv_V = BatchNormalization(axis=ch_axis)(h1_conv_V)

    h2_conv_Y_a = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_Y)
    h2_conv_Y_b = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_Y)
    h2_conv_Y = merge([h2_conv_Y_a, h2_conv_Y_b], mode='max', concat_axis=ch_axis)
    h2_conv_Y = BatchNormalization(axis=ch_axis)(h2_conv_Y)

    h2_conv_U_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_U)
    h2_conv_U_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_U)
    h2_conv_U = merge([h2_conv_U_a, h2_conv_U_b], mode='max', concat_axis=ch_axis)
    h2_conv_U = BatchNormalization(axis=ch_axis)(h2_conv_U)

    h2_conv_V_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_V)
    h2_conv_V_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_V)
    h2_conv_V = merge([h2_conv_V_a, h2_conv_V_b], mode='max', concat_axis=ch_axis)
    h2_conv_V = BatchNormalization(axis=ch_axis)(h2_conv_V)

    h3_conv_Y_a = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_Y)
    h3_conv_Y_b = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_Y)
    h3_conv_Y = merge([h3_conv_Y_a, h3_conv_Y_b], mode='max', concat_axis=ch_axis)
    h3_conv_Y = BatchNormalization(axis=ch_axis)(h3_conv_Y)

    h3_conv_U_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_U)
    h3_conv_U_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_U)
    h3_conv_U = merge([h3_conv_U_a, h3_conv_U_b], mode='max', concat_axis=ch_axis)
    h3_conv_U = BatchNormalization(axis=ch_axis)(h3_conv_U)

    h3_conv_V_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_V)
    h3_conv_V_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_V)
    h3_conv_V = merge([h3_conv_V_a, h3_conv_V_b], mode='max', concat_axis=ch_axis)
    h3_conv_V = BatchNormalization(axis=ch_axis)(h3_conv_V)

    h4_conv_Y_a = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_Y)
    h4_conv_Y_b = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_Y)
    h4_conv_Y = merge([h4_conv_Y_a, h4_conv_Y_b], mode='max', concat_axis=ch_axis)
    h4_conv_Y = BatchNormalization(axis=ch_axis)(h4_conv_Y)

    h4_conv_U_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_U)
    h4_conv_U_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_U)
    h4_conv_U = merge([h4_conv_U_a, h4_conv_U_b], mode='max', concat_axis=ch_axis)
    h4_conv_U = BatchNormalization(axis=ch_axis)(h4_conv_U)

    h4_conv_V_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_V)
    h4_conv_V_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_V)
    h4_conv_V = merge([h4_conv_V_a, h4_conv_V_b], mode='max', concat_axis=ch_axis)
    h4_conv_V = BatchNormalization(axis=ch_axis)(h4_conv_V)

    poolY = MaxPooling2D(pool_size=(2, 2))(h4_conv_Y)
    poolU = MaxPooling2D(pool_size=(2, 2))(h4_conv_U)
    poolV = MaxPooling2D(pool_size=(2, 2))(h4_conv_V)

    poolY = Dropout(0.2)(poolY)
    poolU = Dropout(0.2)(poolU)
    poolV = Dropout(0.2)(poolV)

    # Inline connections
    Y_to_Y_a = Convolution2D(36, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_Y_b = Convolution2D(36, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_Y = merge([Y_to_Y_a, Y_to_Y_b], mode='max', concat_axis=ch_axis)
    Y_to_Y = BatchNormalization(axis=ch_axis)(Y_to_Y)

    U_to_U_a = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_U_b = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_U = merge([U_to_U_a, U_to_U_b], mode='max', concat_axis=ch_axis)
    U_to_U = BatchNormalization(axis=ch_axis)(U_to_U)

    V_to_V_a = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_V_b = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_V = merge([V_to_V_a, V_to_V_b], mode='max', concat_axis=ch_axis)
    V_to_V = BatchNormalization(axis=ch_axis)(Y_to_Y)

    # Cross connections: Y <-> U, Y <-> V
    Y_to_UV_a = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_UV_b = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_UV = merge([Y_to_UV_a, Y_to_UV_b], mode='max', concat_axis=ch_axis)
    Y_to_UV = BatchNormalization(axis=ch_axis)(Y_to_UV)

    U_to_Y_a = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_Y_b = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_Y = merge([U_to_Y_a, U_to_Y_b], mode='max', concat_axis=ch_axis)
    U_to_Y = BatchNormalization(axis=ch_axis)(U_to_Y)

    V_to_Y_a = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_Y_b = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_Y = merge([V_to_Y_a, V_to_Y_b], mode='max', concat_axis=ch_axis)
    V_to_Y = BatchNormalization(axis=ch_axis)(V_to_Y)



    Ymap = merge([Y_to_Y, U_to_Y, V_to_Y], mode='concat', concat_axis=ch_axis)
    Umap = merge([U_to_U, Y_to_UV], mode='concat', concat_axis=ch_axis)
    Vmap = merge([V_to_V, Y_to_UV], mode='concat', concat_axis=ch_axis)

    h5_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h5_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h5_conv_Y = merge([h5_conv_Y_a, h5_conv_Y_b], mode='max', concat_axis=ch_axis)
    h5_conv_Y = BatchNormalization(axis=ch_axis)(h5_conv_Y)

    h5_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h5_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h5_conv_U = merge([h5_conv_U_a, h5_conv_U_b], mode='max', concat_axis=ch_axis)
    h5_conv_U = BatchNormalization(axis=ch_axis)(h5_conv_U)

    h5_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h5_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h5_conv_V = merge([h5_conv_V_a, h5_conv_V_b], mode='max', concat_axis=ch_axis)
    h5_conv_V = BatchNormalization(axis=ch_axis)(h5_conv_V)

    h6_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_Y)
    h6_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_Y)
    h6_conv_Y = merge([h6_conv_Y_a, h6_conv_Y_b], mode='max', concat_axis=ch_axis)
    h6_conv_Y = BatchNormalization(axis=ch_axis)(h6_conv_Y)

    h6_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_U)
    h6_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_U)
    h6_conv_U = merge([h6_conv_U_a, h6_conv_U_b], mode='max', concat_axis=ch_axis)
    h6_conv_U = BatchNormalization(axis=ch_axis)(h6_conv_U)

    h6_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_V)
    h6_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_V)
    h6_conv_V = merge([h6_conv_V_a, h6_conv_V_b], mode='max', concat_axis=ch_axis)
    h6_conv_V = BatchNormalization(axis=ch_axis)(h6_conv_V)

    h7_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_Y)
    h7_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_Y)
    h7_conv_Y = merge([h7_conv_Y_a, h7_conv_Y_b], mode='max', concat_axis=ch_axis)
    h7_conv_Y = BatchNormalization(axis=ch_axis)(h7_conv_Y)

    h7_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_U)
    h7_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_U)
    h7_conv_U = merge([h7_conv_U_a, h7_conv_U_b], mode='max', concat_axis=ch_axis)
    h7_conv_U = BatchNormalization(axis=ch_axis)(h7_conv_U)

    h7_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_V)
    h7_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_V)
    h7_conv_V = merge([h7_conv_V_a, h7_conv_V_b], mode='max', concat_axis=ch_axis)
    h7_conv_V = BatchNormalization(axis=ch_axis)(h7_conv_V)

    h8_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_Y)
    h8_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_Y)
    h8_conv_Y = merge([h8_conv_Y_a, h8_conv_Y_b], mode='max', concat_axis=ch_axis)
    h8_conv_Y = BatchNormalization(axis=ch_axis)(h8_conv_Y)

    h8_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_U)
    h8_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_U)
    h8_conv_U = merge([h8_conv_U_a, h8_conv_U_b], mode='max', concat_axis=ch_axis)
    h8_conv_U = BatchNormalization(axis=ch_axis)(h8_conv_U)

    h8_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_V)
    h8_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_V)
    h8_conv_V = merge([h8_conv_V_a, h8_conv_V_b], mode='max', concat_axis=ch_axis)
    h8_conv_V = BatchNormalization(axis=ch_axis)(h8_conv_V)

    h9_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_Y)
    h9_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_Y)
    h9_conv_Y = merge([h9_conv_Y_a, h9_conv_Y_b], mode='max', concat_axis=ch_axis)
    h9_conv_Y = BatchNormalization(axis=ch_axis)(h9_conv_Y)

    h9_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_U)
    h9_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_U)
    h9_conv_U = merge([h9_conv_U_a, h9_conv_U_b], mode='max', concat_axis=ch_axis)
    h9_conv_U = BatchNormalization(axis=ch_axis)(h9_conv_U)

    h9_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_V)
    h9_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_V)
    h9_conv_V = merge([h9_conv_V_a, h9_conv_V_b], mode='max', concat_axis=ch_axis)
    h9_conv_V = BatchNormalization(axis=ch_axis)(h9_conv_V)

    h10_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_Y)
    h10_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_Y)
    h10_conv_Y = merge([h10_conv_Y_a, h10_conv_Y_b], mode='max', concat_axis=ch_axis)
    h10_conv_Y = BatchNormalization(axis=ch_axis)(h10_conv_Y)

    h10_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_U)
    h10_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_U)
    h10_conv_U = merge([h10_conv_U_a, h10_conv_U_b], mode='max', concat_axis=ch_axis)
    h10_conv_U = BatchNormalization(axis=ch_axis)(h10_conv_U)

    h10_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_V)
    h10_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_V)
    h10_conv_V = merge([h10_conv_V_a, h10_conv_V_b], mode='max', concat_axis=ch_axis)
    h10_conv_V = BatchNormalization(axis=ch_axis)(h10_conv_V)

    poolY = MaxPooling2D(pool_size=(2, 2))(h10_conv_Y)
    poolU = MaxPooling2D(pool_size=(2, 2))(h10_conv_U)
    poolV = MaxPooling2D(pool_size=(2, 2))(h10_conv_V)

    poolY = Dropout(0.2)(poolY)
    poolU = Dropout(0.2)(poolU)
    poolV = Dropout(0.2)(poolV)

    # Inline connections
    Y_to_Y_a = Convolution2D(60, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_Y_b = Convolution2D(60, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_Y = merge([Y_to_Y_a, Y_to_Y_b], mode='max', concat_axis=ch_axis)
    Y_to_Y = BatchNormalization(axis=ch_axis)(Y_to_Y)

    U_to_U_a = Convolution2D(30, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_U_b = Convolution2D(30, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_U = merge([U_to_U_a, U_to_U_b], mode='max', concat_axis=ch_axis)
    U_to_U = BatchNormalization(axis=ch_axis)(U_to_U)

    V_to_V_a = Convolution2D(30, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_V_b = Convolution2D(30, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_V = merge([V_to_V_a, V_to_V_b], mode='max', concat_axis=ch_axis)
    V_to_V = BatchNormalization(axis=ch_axis)(Y_to_Y)

    # Cross connections: Y <-> U, Y <-> V
    Y_to_UV_a = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_UV_b = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_UV = merge([Y_to_UV_a, Y_to_UV_b], mode='max', concat_axis=ch_axis)
    Y_to_UV = BatchNormalization(axis=ch_axis)(Y_to_UV)

    U_to_Y_a = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_Y_b = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_Y = merge([U_to_Y_a, U_to_Y_b], mode='max', concat_axis=ch_axis)
    U_to_Y = BatchNormalization(axis=ch_axis)(U_to_Y)

    V_to_Y_a = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_Y_b = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_Y = merge([V_to_Y_a, V_to_Y_b], mode='max', concat_axis=ch_axis)
    V_to_Y = BatchNormalization(axis=ch_axis)(V_to_Y)

    Ymap = merge([Y_to_Y, U_to_Y, V_to_Y], mode='concat', concat_axis=ch_axis)
    Umap = merge([U_to_U, Y_to_UV], mode='concat', concat_axis=ch_axis)
    Vmap = merge([V_to_V, Y_to_UV], mode='concat', concat_axis=ch_axis)

    h11_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h11_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h11_conv_Y = merge([h11_conv_Y_a, h11_conv_Y_b], mode='max', concat_axis=ch_axis)
    h11_conv_Y = BatchNormalization(axis=ch_axis)(h11_conv_Y)

    h11_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h11_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h11_conv_U = merge([h11_conv_U_a, h11_conv_U_b], mode='max', concat_axis=ch_axis)
    h11_conv_U = BatchNormalization(axis=ch_axis)(h11_conv_U)

    h11_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h11_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h11_conv_V = merge([h11_conv_V_a, h11_conv_V_b], mode='max', concat_axis=ch_axis)
    h11_conv_V = BatchNormalization(axis=ch_axis)(h11_conv_V)

    h12_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_Y)
    h12_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_Y)
    h12_conv_Y = merge([h12_conv_Y_a, h12_conv_Y_b], mode='max', concat_axis=ch_axis)
    h12_conv_Y = BatchNormalization(axis=ch_axis)(h12_conv_Y)

    h12_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_U)
    h12_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_U)
    h12_conv_U = merge([h12_conv_U_a, h12_conv_U_b], mode='max', concat_axis=ch_axis)
    h12_conv_U = BatchNormalization(axis=ch_axis)(h12_conv_U)

    h12_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_V)
    h12_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_V)
    h12_conv_V = merge([h12_conv_V_a, h12_conv_V_b], mode='max', concat_axis=ch_axis)
    h12_conv_V = BatchNormalization(axis=ch_axis)(h12_conv_V)

    h13_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_Y)
    h13_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_Y)
    h13_conv_Y = merge([h13_conv_Y_a, h13_conv_Y_b], mode='max', concat_axis=ch_axis)
    h13_conv_Y = BatchNormalization(axis=ch_axis)(h13_conv_Y)

    h13_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_U)
    h13_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_U)
    h13_conv_U = merge([h13_conv_U_a, h13_conv_U_b], mode='max', concat_axis=ch_axis)
    h13_conv_U = BatchNormalization(axis=ch_axis)(h13_conv_U)

    h13_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_V)
    h13_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_V)
    h13_conv_V = merge([h13_conv_V_a, h13_conv_V_b], mode='max', concat_axis=ch_axis)
    h13_conv_V = BatchNormalization(axis=ch_axis)(h13_conv_V)

    h14_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_Y)
    h14_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_Y)
    h14_conv_Y = merge([h14_conv_Y_a, h14_conv_Y_b], mode='max', concat_axis=ch_axis)
    h14_conv_Y = BatchNormalization(axis=ch_axis)(h14_conv_Y)

    h14_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_U)
    h14_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_U)
    h14_conv_U = merge([h14_conv_U_a, h14_conv_U_b], mode='max', concat_axis=ch_axis)
    h14_conv_U = BatchNormalization(axis=ch_axis)(h14_conv_U)

    h14_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_V)
    h14_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_V)
    h14_conv_V = merge([h14_conv_V_a, h14_conv_V_b], mode='max', concat_axis=ch_axis)
    h14_conv_V = BatchNormalization(axis=ch_axis)(h14_conv_V)

    h15_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_Y)
    h15_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_Y)
    h15_conv_Y = merge([h15_conv_Y_a, h15_conv_Y_b], mode='max', concat_axis=ch_axis)
    h15_conv_Y = BatchNormalization(axis=ch_axis)(h15_conv_Y)

    h15_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_U)
    h15_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_U)
    h15_conv_U = merge([h15_conv_U_a, h15_conv_U_b], mode='max', concat_axis=ch_axis)
    h15_conv_U = BatchNormalization(axis=ch_axis)(h15_conv_U)

    h15_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_V)
    h15_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_V)
    h15_conv_V = merge([h15_conv_V_a, h15_conv_V_b], mode='max', concat_axis=ch_axis)
    h15_conv_V = BatchNormalization(axis=ch_axis)(h15_conv_V)

    h16_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_Y)
    h16_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_Y)
    h16_conv_Y = merge([h16_conv_Y_a, h16_conv_Y_b], mode='max', concat_axis=ch_axis)
    h16_conv_Y = BatchNormalization(axis=ch_axis)(h16_conv_Y)

    h16_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_U)
    h16_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_U)
    h16_conv_U = merge([h16_conv_U_a, h16_conv_U_b], mode='max', concat_axis=ch_axis)
    h16_conv_U = BatchNormalization(axis=ch_axis)(h16_conv_U)

    h16_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_V)
    h16_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_V)
    h16_conv_V = merge([h16_conv_V_a, h16_conv_V_b], mode='max', concat_axis=ch_axis)
    h16_conv_V = BatchNormalization(axis=ch_axis)(h16_conv_V)

    poolY = MaxPooling2D(pool_size=(8, 8))(h16_conv_Y)
    poolU = MaxPooling2D(pool_size=(8, 8))(h16_conv_U)
    poolV = MaxPooling2D(pool_size=(8, 8))(h16_conv_V)

    poolY = Dropout(0.2)(poolY)
    poolU = Dropout(0.2)(poolU)
    poolV = Dropout(0.2)(poolV)

    concat_map = merge([poolY, poolU, poolV], mode='concat', concat_axis=ch_axis)

    h16 = Flatten()(concat_map)
    h17 = MaxoutDense(500, nb_feature=5, init='glorot_uniform', W_regularizer=l2(0.0005))(h16)
    h17 = BatchNormalization(axis=1)(h17)
    h17_drop = Dropout(0.2)(h17)
    out = Dense(nb_classes, activation='softmax', init='glorot_uniform', W_regularizer=l2(0.0005))(h17)

    model = Model(input=inputYUV, output=out)
    return model


def get_xfitnet_elu_con(nb_classes, input_shape, ch_axis):
    inputYUV = Input(shape=input_shape)
    inputNorm = BatchNormalization(axis=ch_axis)(inputYUV)

    inp = inputNorm
    if K.image_dim_ordering() == 'tf':
        lb1 = Lambda(lambda x: x[:, :, :, 0:1], output_shape=(32, 32, 1))(inp)
        lb2 = Lambda(lambda x: x[:, :, :, 1:2], output_shape=(32, 32, 1))(inp)
        lb3 = Lambda(lambda x: x[:, :, :, 2:3], output_shape=(32, 32, 1))(inp)
    else:
        lb1 = Lambda(lambda x: x[:, 0:1, :, :], output_shape=(1, 32, 32))(inp)
        lb2 = Lambda(lambda x: x[:, 1:2, :, :], output_shape=(1, 32, 32))(inp)
        lb3 = Lambda(lambda x: x[:, 2:3, :, :], output_shape=(1, 32, 32))(inp)

    inputY = lb1
    inputU = lb2
    inputV = lb3

    inputY_drop = Dropout(0.2)(inputY)
    inputU_drop = Dropout(0.2)(inputU)
    inputV_drop = Dropout(0.2)(inputV)

    h0_conv_Y_a = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputY)
    h0_conv_Y_b = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputY)
    h0_conv_Y = merge([h0_conv_Y_a, h0_conv_Y_b], mode='max', concat_axis=ch_axis)
    h0_conv_Y = BatchNormalization(axis=ch_axis)(h0_conv_Y)

    h0_conv_U_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputU)
    h0_conv_U_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputU)
    h0_conv_U = merge([h0_conv_U_a, h0_conv_U_b], mode='max', concat_axis=ch_axis)
    h0_conv_U = BatchNormalization(axis=ch_axis)(h0_conv_U)

    h0_conv_V_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputV)
    h0_conv_V_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputV)
    h0_conv_V = merge([h0_conv_V_a, h0_conv_V_b], mode='max', concat_axis=ch_axis)
    h0_conv_V = BatchNormalization(axis=ch_axis)(h0_conv_V)

    h1_conv_Y_a = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_Y)
    h1_conv_Y_b = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_Y)
    h1_conv_Y = merge([h1_conv_Y_a, h1_conv_Y_b], mode='max', concat_axis=ch_axis)
    h1_conv_Y = BatchNormalization(axis=ch_axis)(h1_conv_Y)

    h1_conv_U_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_U)
    h1_conv_U_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_U)
    h1_conv_U = merge([h1_conv_U_a, h1_conv_U_b], mode='max', concat_axis=ch_axis)
    h1_conv_U = BatchNormalization(axis=ch_axis)(h1_conv_U)

    h1_conv_V_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_V)
    h1_conv_V_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_V)
    h1_conv_V = merge([h1_conv_V_a, h1_conv_V_b], mode='max', concat_axis=ch_axis)
    h1_conv_V = BatchNormalization(axis=ch_axis)(h1_conv_V)

    h2_conv_Y_a = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_Y)
    h2_conv_Y_b = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_Y)
    h2_conv_Y = merge([h2_conv_Y_a, h2_conv_Y_b], mode='max', concat_axis=ch_axis)
    h2_conv_Y = BatchNormalization(axis=ch_axis)(h2_conv_Y)

    h2_conv_U_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_U)
    h2_conv_U_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_U)
    h2_conv_U = merge([h2_conv_U_a, h2_conv_U_b], mode='max', concat_axis=ch_axis)
    h2_conv_U = BatchNormalization(axis=ch_axis)(h2_conv_U)

    h2_conv_V_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_V)
    h2_conv_V_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_V)
    h2_conv_V = merge([h2_conv_V_a, h2_conv_V_b], mode='max', concat_axis=ch_axis)
    h2_conv_V = BatchNormalization(axis=ch_axis)(h2_conv_V)

    h3_conv_Y_a = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_Y)
    h3_conv_Y_b = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_Y)
    h3_conv_Y = merge([h3_conv_Y_a, h3_conv_Y_b], mode='max', concat_axis=ch_axis)
    h3_conv_Y = BatchNormalization(axis=ch_axis)(h3_conv_Y)

    h3_conv_U_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_U)
    h3_conv_U_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_U)
    h3_conv_U = merge([h3_conv_U_a, h3_conv_U_b], mode='max', concat_axis=ch_axis)
    h3_conv_U = BatchNormalization(axis=ch_axis)(h3_conv_U)

    h3_conv_V_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_V)
    h3_conv_V_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_V)
    h3_conv_V = merge([h3_conv_V_a, h3_conv_V_b], mode='max', concat_axis=ch_axis)
    h3_conv_V = BatchNormalization(axis=ch_axis)(h3_conv_V)

    h4_conv_Y_a = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_Y)
    h4_conv_Y_b = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_Y)
    h4_conv_Y = merge([h4_conv_Y_a, h4_conv_Y_b], mode='max', concat_axis=ch_axis)
    h4_conv_Y = BatchNormalization(axis=ch_axis)(h4_conv_Y)

    h4_conv_U_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_U)
    h4_conv_U_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_U)
    h4_conv_U = merge([h4_conv_U_a, h4_conv_U_b], mode='max', concat_axis=ch_axis)
    h4_conv_U = BatchNormalization(axis=ch_axis)(h4_conv_U)

    h4_conv_V_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_V)
    h4_conv_V_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_V)
    h4_conv_V = merge([h4_conv_V_a, h4_conv_V_b], mode='max', concat_axis=ch_axis)
    h4_conv_V = BatchNormalization(axis=ch_axis)(h4_conv_V)

    poolY = MaxPooling2D(pool_size=(2, 2))(h4_conv_Y)
    poolU = MaxPooling2D(pool_size=(2, 2))(h4_conv_U)
    poolV = MaxPooling2D(pool_size=(2, 2))(h4_conv_V)

    poolY = Dropout(0.2)(poolY)
    poolU = Dropout(0.2)(poolU)
    poolV = Dropout(0.2)(poolV)

    # Inline connections
    Y_to_Y_a = Convolution2D(36, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_Y_b = Convolution2D(36, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_Y = merge([Y_to_Y_a, Y_to_Y_b], mode='max', concat_axis=ch_axis)
    Y_to_Y = BatchNormalization(axis=ch_axis)(Y_to_Y)

    U_to_U_a = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_U_b = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_U = merge([U_to_U_a, U_to_U_b], mode='max', concat_axis=ch_axis)
    U_to_U = BatchNormalization(axis=ch_axis)(U_to_U)

    V_to_V_a = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_V_b = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_V = merge([V_to_V_a, V_to_V_b], mode='max', concat_axis=ch_axis)
    V_to_V = BatchNormalization(axis=ch_axis)(Y_to_Y)

    # Cross connections: Y <-> U, Y <-> V
    Y_to_UV_a = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_UV_b = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_UV = merge([Y_to_UV_a, Y_to_UV_b], mode='max', concat_axis=ch_axis)
    Y_to_UV = BatchNormalization(axis=ch_axis)(Y_to_UV)

    U_to_Y_a = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_Y_b = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_Y = merge([U_to_Y_a, U_to_Y_b], mode='max', concat_axis=ch_axis)
    U_to_Y = BatchNormalization(axis=ch_axis)(U_to_Y)

    V_to_Y_a = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_Y_b = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_Y = merge([V_to_Y_a, V_to_Y_b], mode='max', concat_axis=ch_axis)
    V_to_Y = BatchNormalization(axis=ch_axis)(V_to_Y)

    Ymap = merge([Y_to_Y, U_to_Y, V_to_Y], mode='concat', concat_axis=ch_axis)
    Umap = merge([U_to_U, Y_to_UV], mode='concat', concat_axis=ch_axis)
    Vmap = merge([V_to_V, Y_to_UV], mode='concat', concat_axis=ch_axis)

    h5_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h5_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h5_conv_Y = merge([h5_conv_Y_a, h5_conv_Y_b], mode='max', concat_axis=ch_axis)
    h5_conv_Y = BatchNormalization(axis=ch_axis)(h5_conv_Y)

    h5_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h5_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h5_conv_U = merge([h5_conv_U_a, h5_conv_U_b], mode='max', concat_axis=ch_axis)
    h5_conv_U = BatchNormalization(axis=ch_axis)(h5_conv_U)

    h5_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h5_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h5_conv_V = merge([h5_conv_V_a, h5_conv_V_b], mode='max', concat_axis=ch_axis)
    h5_conv_V = BatchNormalization(axis=ch_axis)(h5_conv_V)

    h6_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_Y)
    h6_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_Y)
    h6_conv_Y = merge([h6_conv_Y_a, h6_conv_Y_b], mode='max', concat_axis=ch_axis)
    h6_conv_Y = BatchNormalization(axis=ch_axis)(h6_conv_Y)

    h6_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_U)
    h6_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_U)
    h6_conv_U = merge([h6_conv_U_a, h6_conv_U_b], mode='max', concat_axis=ch_axis)
    h6_conv_U = BatchNormalization(axis=ch_axis)(h6_conv_U)

    h6_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_V)
    h6_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_V)
    h6_conv_V = merge([h6_conv_V_a, h6_conv_V_b], mode='max', concat_axis=ch_axis)
    h6_conv_V = BatchNormalization(axis=ch_axis)(h6_conv_V)

    h7_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_Y)
    h7_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_Y)
    h7_conv_Y = merge([h7_conv_Y_a, h7_conv_Y_b], mode='max', concat_axis=ch_axis)
    h7_conv_Y = BatchNormalization(axis=ch_axis)(h7_conv_Y)

    h7_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_U)
    h7_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_U)
    h7_conv_U = merge([h7_conv_U_a, h7_conv_U_b], mode='max', concat_axis=ch_axis)
    h7_conv_U = BatchNormalization(axis=ch_axis)(h7_conv_U)

    h7_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_V)
    h7_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_V)
    h7_conv_V = merge([h7_conv_V_a, h7_conv_V_b], mode='max', concat_axis=ch_axis)
    h7_conv_V = BatchNormalization(axis=ch_axis)(h7_conv_V)

    h8_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_Y)
    h8_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_Y)
    h8_conv_Y = merge([h8_conv_Y_a, h8_conv_Y_b], mode='max', concat_axis=ch_axis)
    h8_conv_Y = BatchNormalization(axis=ch_axis)(h8_conv_Y)

    h8_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_U)
    h8_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_U)
    h8_conv_U = merge([h8_conv_U_a, h8_conv_U_b], mode='max', concat_axis=ch_axis)
    h8_conv_U = BatchNormalization(axis=ch_axis)(h8_conv_U)

    h8_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_V)
    h8_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_V)
    h8_conv_V = merge([h8_conv_V_a, h8_conv_V_b], mode='max', concat_axis=ch_axis)
    h8_conv_V = BatchNormalization(axis=ch_axis)(h8_conv_V)

    h9_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_Y)
    h9_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_Y)
    h9_conv_Y = merge([h9_conv_Y_a, h9_conv_Y_b], mode='max', concat_axis=ch_axis)
    h9_conv_Y = BatchNormalization(axis=ch_axis)(h9_conv_Y)

    h9_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_U)
    h9_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_U)
    h9_conv_U = merge([h9_conv_U_a, h9_conv_U_b], mode='max', concat_axis=ch_axis)
    h9_conv_U = BatchNormalization(axis=ch_axis)(h9_conv_U)

    h9_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_V)
    h9_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_V)
    h9_conv_V = merge([h9_conv_V_a, h9_conv_V_b], mode='max', concat_axis=ch_axis)
    h9_conv_V = BatchNormalization(axis=ch_axis)(h9_conv_V)

    h10_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_Y)
    h10_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_Y)
    h10_conv_Y = merge([h10_conv_Y_a, h10_conv_Y_b], mode='max', concat_axis=ch_axis)
    h10_conv_Y = BatchNormalization(axis=ch_axis)(h10_conv_Y)

    h10_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_U)
    h10_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_U)
    h10_conv_U = merge([h10_conv_U_a, h10_conv_U_b], mode='max', concat_axis=ch_axis)
    h10_conv_U = BatchNormalization(axis=ch_axis)(h10_conv_U)

    h10_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_V)
    h10_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_V)
    h10_conv_V = merge([h10_conv_V_a, h10_conv_V_b], mode='max', concat_axis=ch_axis)
    h10_conv_V = BatchNormalization(axis=ch_axis)(h10_conv_V)

    poolY = MaxPooling2D(pool_size=(2, 2))(h10_conv_Y)
    poolU = MaxPooling2D(pool_size=(2, 2))(h10_conv_U)
    poolV = MaxPooling2D(pool_size=(2, 2))(h10_conv_V)

    poolY = Dropout(0.2)(poolY)
    poolU = Dropout(0.2)(poolU)
    poolV = Dropout(0.2)(poolV)

    U_to_Y = Convolution2D(30, 1, 1, border_mode='same', activation='elu')(poolU)
    # U_to_Y = Convolution2D(8, 1, 1, border_mode='same', activation='relu')(U_to_Y)
    U_to_V = Convolution2D(30, 1, 1, border_mode='same', activation='elu')(poolU)
    # U_to_V = Convolution2D(8, 1, 1, border_mode='same', activation='relu')(U_to_V)

    V_to_Y = Convolution2D(30, 1, 1, border_mode='same', activation='elu')(poolV)
    # V_to_Y = Convolution2D(8, 1, 1, border_mode='same', activation='relu')(V_to_Y)
    V_to_U = Convolution2D(30, 1, 1, border_mode='same', activation='elu')(poolV)
    # V_to_U = Convolution2D(8, 1, 1, border_mode='same', activation='relu')(V_to_U)

    Y_to_U = Convolution2D(60, 1, 1, border_mode='same', activation='elu')(poolY)
    # Y_to_U = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(Y_to_U)
    Y_to_V = Convolution2D(60, 1, 1, border_mode='same', activation='elu')(poolY)
    # Y_to_V = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(Y_to_V)

    Ymap = merge([poolY, U_to_Y, V_to_Y], mode='concat', concat_axis=ch_axis)
    Umap = merge([poolU, Y_to_U, V_to_U], mode='concat', concat_axis=ch_axis)
    Vmap = merge([poolV, Y_to_V, U_to_V], mode='concat', concat_axis=ch_axis)

    h11_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h11_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h11_conv_Y = merge([h11_conv_Y_a, h11_conv_Y_b], mode='max', concat_axis=ch_axis)
    h11_conv_Y = BatchNormalization(axis=ch_axis)(h11_conv_Y)

    h11_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h11_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h11_conv_U = merge([h11_conv_U_a, h11_conv_U_b], mode='max', concat_axis=ch_axis)
    h11_conv_U = BatchNormalization(axis=ch_axis)(h11_conv_U)

    h11_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h11_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h11_conv_V = merge([h11_conv_V_a, h11_conv_V_b], mode='max', concat_axis=ch_axis)
    h11_conv_V = BatchNormalization(axis=ch_axis)(h11_conv_V)

    h12_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_Y)
    h12_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_Y)
    h12_conv_Y = merge([h12_conv_Y_a, h12_conv_Y_b], mode='max', concat_axis=ch_axis)
    h12_conv_Y = BatchNormalization(axis=ch_axis)(h12_conv_Y)

    h12_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_U)
    h12_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_U)
    h12_conv_U = merge([h12_conv_U_a, h12_conv_U_b], mode='max', concat_axis=ch_axis)
    h12_conv_U = BatchNormalization(axis=ch_axis)(h12_conv_U)

    h12_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_V)
    h12_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_V)
    h12_conv_V = merge([h12_conv_V_a, h12_conv_V_b], mode='max', concat_axis=ch_axis)
    h12_conv_V = BatchNormalization(axis=ch_axis)(h12_conv_V)

    h13_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_Y)
    h13_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_Y)
    h13_conv_Y = merge([h13_conv_Y_a, h13_conv_Y_b], mode='max', concat_axis=ch_axis)
    h13_conv_Y = BatchNormalization(axis=ch_axis)(h13_conv_Y)

    h13_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_U)
    h13_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_U)
    h13_conv_U = merge([h13_conv_U_a, h13_conv_U_b], mode='max', concat_axis=ch_axis)
    h13_conv_U = BatchNormalization(axis=ch_axis)(h13_conv_U)

    h13_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_V)
    h13_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_V)
    h13_conv_V = merge([h13_conv_V_a, h13_conv_V_b], mode='max', concat_axis=ch_axis)
    h13_conv_V = BatchNormalization(axis=ch_axis)(h13_conv_V)

    h14_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_Y)
    h14_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_Y)
    h14_conv_Y = merge([h14_conv_Y_a, h14_conv_Y_b], mode='max', concat_axis=ch_axis)
    h14_conv_Y = BatchNormalization(axis=ch_axis)(h14_conv_Y)

    h14_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_U)
    h14_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_U)
    h14_conv_U = merge([h14_conv_U_a, h14_conv_U_b], mode='max', concat_axis=ch_axis)
    h14_conv_U = BatchNormalization(axis=ch_axis)(h14_conv_U)

    h14_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_V)
    h14_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_V)
    h14_conv_V = merge([h14_conv_V_a, h14_conv_V_b], mode='max', concat_axis=ch_axis)
    h14_conv_V = BatchNormalization(axis=ch_axis)(h14_conv_V)

    h15_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_Y)
    h15_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_Y)
    h15_conv_Y = merge([h15_conv_Y_a, h15_conv_Y_b], mode='max', concat_axis=ch_axis)
    h15_conv_Y = BatchNormalization(axis=ch_axis)(h15_conv_Y)

    h15_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_U)
    h15_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_U)
    h15_conv_U = merge([h15_conv_U_a, h15_conv_U_b], mode='max', concat_axis=ch_axis)
    h15_conv_U = BatchNormalization(axis=ch_axis)(h15_conv_U)

    h15_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_V)
    h15_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_V)
    h15_conv_V = merge([h15_conv_V_a, h15_conv_V_b], mode='max', concat_axis=ch_axis)
    h15_conv_V = BatchNormalization(axis=ch_axis)(h15_conv_V)

    h16_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_Y)
    h16_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_Y)
    h16_conv_Y = merge([h16_conv_Y_a, h16_conv_Y_b], mode='max', concat_axis=ch_axis)
    h16_conv_Y = BatchNormalization(axis=ch_axis)(h16_conv_Y)

    h16_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_U)
    h16_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_U)
    h16_conv_U = merge([h16_conv_U_a, h16_conv_U_b], mode='max', concat_axis=ch_axis)
    h16_conv_U = BatchNormalization(axis=ch_axis)(h16_conv_U)

    h16_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_V)
    h16_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_V)
    h16_conv_V = merge([h16_conv_V_a, h16_conv_V_b], mode='max', concat_axis=ch_axis)
    h16_conv_V = BatchNormalization(axis=ch_axis)(h16_conv_V)

    poolY = MaxPooling2D(pool_size=(8, 8))(h16_conv_Y)
    poolU = MaxPooling2D(pool_size=(8, 8))(h16_conv_U)
    poolV = MaxPooling2D(pool_size=(8, 8))(h16_conv_V)

    poolY = Dropout(0.2)(poolY)
    poolU = Dropout(0.2)(poolU)
    poolV = Dropout(0.2)(poolV)

    concat_map = merge([poolY, poolU, poolV], mode='concat', concat_axis=ch_axis)

    h16 = Flatten()(concat_map)
    h17 = MaxoutDense(500, nb_feature=5, init='glorot_uniform', W_regularizer=l2(0.0005))(h16)
    h17 = BatchNormalization(axis=1)(h17)
    h17_drop = Dropout(0.2)(h17)
    out = Dense(nb_classes, activation='softmax', init='glorot_uniform', W_regularizer=l2(0.0005))(h17)

    model = Model(input=inputYUV, output=out)
    return model


def get_xfitnet_nincon1(nb_classes, input_shape, ch_axis):
    inputYUV = Input(shape=input_shape)
    inputNorm = BatchNormalization(axis=ch_axis)(inputYUV)

    inp = inputNorm
    if K.image_dim_ordering() == 'tf':
        lb1 = Lambda(lambda x: x[:, :, :, 0:1], output_shape=(32, 32, 1))(inp)
        lb2 = Lambda(lambda x: x[:, :, :, 1:2], output_shape=(32, 32, 1))(inp)
        lb3 = Lambda(lambda x: x[:, :, :, 2:3], output_shape=(32, 32, 1))(inp)
    else:
        lb1 = Lambda(lambda x: x[:, 0:1, :, :], output_shape=(1, 32, 32))(inp)
        lb2 = Lambda(lambda x: x[:, 1:2, :, :], output_shape=(1, 32, 32))(inp)
        lb3 = Lambda(lambda x: x[:, 2:3, :, :], output_shape=(1, 32, 32))(inp)

    inputY = lb1
    inputU = lb2
    inputV = lb3

    inputY_drop = Dropout(0.2)(inputY)
    inputU_drop = Dropout(0.2)(inputU)
    inputV_drop = Dropout(0.2)(inputV)

    h0_conv_Y_a = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputY)
    h0_conv_Y_b = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputY)
    h0_conv_Y = merge([h0_conv_Y_a, h0_conv_Y_b], mode='max', concat_axis=ch_axis)
    h0_conv_Y = BatchNormalization(axis=ch_axis)(h0_conv_Y)

    h0_conv_U_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputU)
    h0_conv_U_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputU)
    h0_conv_U = merge([h0_conv_U_a, h0_conv_U_b], mode='max', concat_axis=ch_axis)
    h0_conv_U = BatchNormalization(axis=ch_axis)(h0_conv_U)

    h0_conv_V_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputV)
    h0_conv_V_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputV)
    h0_conv_V = merge([h0_conv_V_a, h0_conv_V_b], mode='max', concat_axis=ch_axis)
    h0_conv_V = BatchNormalization(axis=ch_axis)(h0_conv_V)

    h1_conv_Y_a = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_Y)
    h1_conv_Y_b = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_Y)
    h1_conv_Y = merge([h1_conv_Y_a, h1_conv_Y_b], mode='max', concat_axis=ch_axis)
    h1_conv_Y = BatchNormalization(axis=ch_axis)(h1_conv_Y)

    h1_conv_U_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_U)
    h1_conv_U_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_U)
    h1_conv_U = merge([h1_conv_U_a, h1_conv_U_b], mode='max', concat_axis=ch_axis)
    h1_conv_U = BatchNormalization(axis=ch_axis)(h1_conv_U)

    h1_conv_V_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_V)
    h1_conv_V_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_V)
    h1_conv_V = merge([h1_conv_V_a, h1_conv_V_b], mode='max', concat_axis=ch_axis)
    h1_conv_V = BatchNormalization(axis=ch_axis)(h1_conv_V)

    h2_conv_Y_a = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_Y)
    h2_conv_Y_b = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_Y)
    h2_conv_Y = merge([h2_conv_Y_a, h2_conv_Y_b], mode='max', concat_axis=ch_axis)
    h2_conv_Y = BatchNormalization(axis=ch_axis)(h2_conv_Y)

    h2_conv_U_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_U)
    h2_conv_U_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_U)
    h2_conv_U = merge([h2_conv_U_a, h2_conv_U_b], mode='max', concat_axis=ch_axis)
    h2_conv_U = BatchNormalization(axis=ch_axis)(h2_conv_U)

    h2_conv_V_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_V)
    h2_conv_V_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_V)
    h2_conv_V = merge([h2_conv_V_a, h2_conv_V_b], mode='max', concat_axis=ch_axis)
    h2_conv_V = BatchNormalization(axis=ch_axis)(h2_conv_V)

    h3_conv_Y_a = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_Y)
    h3_conv_Y_b = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_Y)
    h3_conv_Y = merge([h3_conv_Y_a, h3_conv_Y_b], mode='max', concat_axis=ch_axis)
    h3_conv_Y = BatchNormalization(axis=ch_axis)(h3_conv_Y)

    h3_conv_U_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_U)
    h3_conv_U_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_U)
    h3_conv_U = merge([h3_conv_U_a, h3_conv_U_b], mode='max', concat_axis=ch_axis)
    h3_conv_U = BatchNormalization(axis=ch_axis)(h3_conv_U)

    h3_conv_V_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_V)
    h3_conv_V_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_V)
    h3_conv_V = merge([h3_conv_V_a, h3_conv_V_b], mode='max', concat_axis=ch_axis)
    h3_conv_V = BatchNormalization(axis=ch_axis)(h3_conv_V)

    h4_conv_Y_a = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_Y)
    h4_conv_Y_b = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_Y)
    h4_conv_Y = merge([h4_conv_Y_a, h4_conv_Y_b], mode='max', concat_axis=ch_axis)
    h4_conv_Y = BatchNormalization(axis=ch_axis)(h4_conv_Y)

    h4_conv_U_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_U)
    h4_conv_U_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_U)
    h4_conv_U = merge([h4_conv_U_a, h4_conv_U_b], mode='max', concat_axis=ch_axis)
    h4_conv_U = BatchNormalization(axis=ch_axis)(h4_conv_U)

    h4_conv_V_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_V)
    h4_conv_V_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_V)
    h4_conv_V = merge([h4_conv_V_a, h4_conv_V_b], mode='max', concat_axis=ch_axis)
    h4_conv_V = BatchNormalization(axis=ch_axis)(h4_conv_V)

    poolY = MaxPooling2D(pool_size=(2, 2))(h4_conv_Y)
    poolU = MaxPooling2D(pool_size=(2, 2))(h4_conv_U)
    poolV = MaxPooling2D(pool_size=(2, 2))(h4_conv_V)

    poolY = Dropout(0.2)(poolY)
    poolU = Dropout(0.2)(poolU)
    poolV = Dropout(0.2)(poolV)

    # Inline connections
    Y_to_Y_a = Convolution2D(36, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_Y_b = Convolution2D(36, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_Y = merge([Y_to_Y_a, Y_to_Y_b], mode='max', concat_axis=ch_axis)
    Y_to_Y = BatchNormalization(axis=ch_axis)(Y_to_Y)

    U_to_U_a = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_U_b = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_U = merge([U_to_U_a, U_to_U_b], mode='max', concat_axis=ch_axis)
    U_to_U = BatchNormalization(axis=ch_axis)(U_to_U)

    V_to_V_a = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_V_b = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_V = merge([V_to_V_a, V_to_V_b], mode='max', concat_axis=ch_axis)
    V_to_V = BatchNormalization(axis=ch_axis)(Y_to_Y)

    # Cross connections: Y <-> U, Y <-> V
    Y_to_UV_a = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_UV_b = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_UV = merge([Y_to_UV_a, Y_to_UV_b], mode='max', concat_axis=ch_axis)
    Y_to_UV = BatchNormalization(axis=ch_axis)(Y_to_UV)

    U_to_Y_a = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_Y_b = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_Y = merge([U_to_Y_a, U_to_Y_b], mode='max', concat_axis=ch_axis)
    U_to_Y = BatchNormalization(axis=ch_axis)(U_to_Y)

    V_to_Y_a = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_Y_b = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_Y = merge([V_to_Y_a, V_to_Y_b], mode='max', concat_axis=ch_axis)
    V_to_Y = BatchNormalization(axis=ch_axis)(V_to_Y)

    Ymap = merge([Y_to_Y, U_to_Y, V_to_Y], mode='concat', concat_axis=ch_axis)
    Umap = merge([U_to_U, Y_to_UV], mode='concat', concat_axis=ch_axis)
    Vmap = merge([V_to_V, Y_to_UV], mode='concat', concat_axis=ch_axis)

    h5_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h5_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h5_conv_Y = merge([h5_conv_Y_a, h5_conv_Y_b], mode='max', concat_axis=ch_axis)
    h5_conv_Y = BatchNormalization(axis=ch_axis)(h5_conv_Y)

    h5_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h5_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h5_conv_U = merge([h5_conv_U_a, h5_conv_U_b], mode='max', concat_axis=ch_axis)
    h5_conv_U = BatchNormalization(axis=ch_axis)(h5_conv_U)

    h5_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h5_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h5_conv_V = merge([h5_conv_V_a, h5_conv_V_b], mode='max', concat_axis=ch_axis)
    h5_conv_V = BatchNormalization(axis=ch_axis)(h5_conv_V)

    h6_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_Y)
    h6_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_Y)
    h6_conv_Y = merge([h6_conv_Y_a, h6_conv_Y_b], mode='max', concat_axis=ch_axis)
    h6_conv_Y = BatchNormalization(axis=ch_axis)(h6_conv_Y)

    h6_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_U)
    h6_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_U)
    h6_conv_U = merge([h6_conv_U_a, h6_conv_U_b], mode='max', concat_axis=ch_axis)
    h6_conv_U = BatchNormalization(axis=ch_axis)(h6_conv_U)

    h6_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_V)
    h6_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_V)
    h6_conv_V = merge([h6_conv_V_a, h6_conv_V_b], mode='max', concat_axis=ch_axis)
    h6_conv_V = BatchNormalization(axis=ch_axis)(h6_conv_V)

    h7_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_Y)
    h7_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_Y)
    h7_conv_Y = merge([h7_conv_Y_a, h7_conv_Y_b], mode='max', concat_axis=ch_axis)
    h7_conv_Y = BatchNormalization(axis=ch_axis)(h7_conv_Y)

    h7_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_U)
    h7_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_U)
    h7_conv_U = merge([h7_conv_U_a, h7_conv_U_b], mode='max', concat_axis=ch_axis)
    h7_conv_U = BatchNormalization(axis=ch_axis)(h7_conv_U)

    h7_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_V)
    h7_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_V)
    h7_conv_V = merge([h7_conv_V_a, h7_conv_V_b], mode='max', concat_axis=ch_axis)
    h7_conv_V = BatchNormalization(axis=ch_axis)(h7_conv_V)

    h8_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_Y)
    h8_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_Y)
    h8_conv_Y = merge([h8_conv_Y_a, h8_conv_Y_b], mode='max', concat_axis=ch_axis)
    h8_conv_Y = BatchNormalization(axis=ch_axis)(h8_conv_Y)

    h8_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_U)
    h8_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_U)
    h8_conv_U = merge([h8_conv_U_a, h8_conv_U_b], mode='max', concat_axis=ch_axis)
    h8_conv_U = BatchNormalization(axis=ch_axis)(h8_conv_U)

    h8_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_V)
    h8_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_V)
    h8_conv_V = merge([h8_conv_V_a, h8_conv_V_b], mode='max', concat_axis=ch_axis)
    h8_conv_V = BatchNormalization(axis=ch_axis)(h8_conv_V)

    h9_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_Y)
    h9_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_Y)
    h9_conv_Y = merge([h9_conv_Y_a, h9_conv_Y_b], mode='max', concat_axis=ch_axis)
    h9_conv_Y = BatchNormalization(axis=ch_axis)(h9_conv_Y)

    h9_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_U)
    h9_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_U)
    h9_conv_U = merge([h9_conv_U_a, h9_conv_U_b], mode='max', concat_axis=ch_axis)
    h9_conv_U = BatchNormalization(axis=ch_axis)(h9_conv_U)

    h9_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_V)
    h9_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_V)
    h9_conv_V = merge([h9_conv_V_a, h9_conv_V_b], mode='max', concat_axis=ch_axis)
    h9_conv_V = BatchNormalization(axis=ch_axis)(h9_conv_V)

    h10_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_Y)
    h10_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_Y)
    h10_conv_Y = merge([h10_conv_Y_a, h10_conv_Y_b], mode='max', concat_axis=ch_axis)
    h10_conv_Y = BatchNormalization(axis=ch_axis)(h10_conv_Y)

    h10_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_U)
    h10_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_U)
    h10_conv_U = merge([h10_conv_U_a, h10_conv_U_b], mode='max', concat_axis=ch_axis)
    h10_conv_U = BatchNormalization(axis=ch_axis)(h10_conv_U)

    h10_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_V)
    h10_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_V)
    h10_conv_V = merge([h10_conv_V_a, h10_conv_V_b], mode='max', concat_axis=ch_axis)
    h10_conv_V = BatchNormalization(axis=ch_axis)(h10_conv_V)

    poolY = MaxPooling2D(pool_size=(2, 2))(h10_conv_Y)
    poolU = MaxPooling2D(pool_size=(2, 2))(h10_conv_U)
    poolV = MaxPooling2D(pool_size=(2, 2))(h10_conv_V)

    poolY = Dropout(0.2)(poolY)
    poolU = Dropout(0.2)(poolU)
    poolV = Dropout(0.2)(poolV)

    U_to_Y = Convolution2D(30, 1, 1, border_mode='same', activation='relu')(poolU)
    U_to_Y = Convolution2D(15, 1, 1, border_mode='same', activation='relu')(U_to_Y)
    U_to_V = Convolution2D(30, 1, 1, border_mode='same', activation='relu')(poolU)
    U_to_V = Convolution2D(15, 1, 1, border_mode='same', activation='relu')(U_to_V)

    V_to_Y = Convolution2D(30, 1, 1, border_mode='same', activation='relu')(poolV)
    V_to_Y = Convolution2D(15, 1, 1, border_mode='same', activation='relu')(V_to_Y)
    V_to_U = Convolution2D(30, 1, 1, border_mode='same', activation='relu')(poolV)
    V_to_U = Convolution2D(15, 1, 1, border_mode='same', activation='relu')(V_to_U)

    Y_to_U = Convolution2D(60, 1, 1, border_mode='same', activation='relu')(poolY)
    Y_to_U = Convolution2D(30, 1, 1, border_mode='same', activation='relu')(Y_to_U)
    Y_to_V = Convolution2D(60, 1, 1, border_mode='same', activation='relu')(poolY)
    Y_to_V = Convolution2D(30, 1, 1, border_mode='same', activation='relu')(Y_to_V)

    Ymap = merge([poolY, U_to_Y, V_to_Y], mode='concat', concat_axis=ch_axis)
    Umap = merge([poolU, Y_to_U, V_to_U], mode='concat', concat_axis=ch_axis)
    Vmap = merge([poolV, Y_to_V, U_to_V], mode='concat', concat_axis=ch_axis)

    h11_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h11_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h11_conv_Y = merge([h11_conv_Y_a, h11_conv_Y_b], mode='max', concat_axis=ch_axis)
    h11_conv_Y = BatchNormalization(axis=ch_axis)(h11_conv_Y)

    h11_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h11_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h11_conv_U = merge([h11_conv_U_a, h11_conv_U_b], mode='max', concat_axis=ch_axis)
    h11_conv_U = BatchNormalization(axis=ch_axis)(h11_conv_U)

    h11_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h11_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h11_conv_V = merge([h11_conv_V_a, h11_conv_V_b], mode='max', concat_axis=ch_axis)
    h11_conv_V = BatchNormalization(axis=ch_axis)(h11_conv_V)

    h12_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_Y)
    h12_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_Y)
    h12_conv_Y = merge([h12_conv_Y_a, h12_conv_Y_b], mode='max', concat_axis=ch_axis)
    h12_conv_Y = BatchNormalization(axis=ch_axis)(h12_conv_Y)

    h12_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_U)
    h12_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_U)
    h12_conv_U = merge([h12_conv_U_a, h12_conv_U_b], mode='max', concat_axis=ch_axis)
    h12_conv_U = BatchNormalization(axis=ch_axis)(h12_conv_U)

    h12_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_V)
    h12_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_V)
    h12_conv_V = merge([h12_conv_V_a, h12_conv_V_b], mode='max', concat_axis=ch_axis)
    h12_conv_V = BatchNormalization(axis=ch_axis)(h12_conv_V)

    h13_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_Y)
    h13_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_Y)
    h13_conv_Y = merge([h13_conv_Y_a, h13_conv_Y_b], mode='max', concat_axis=ch_axis)
    h13_conv_Y = BatchNormalization(axis=ch_axis)(h13_conv_Y)

    h13_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_U)
    h13_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_U)
    h13_conv_U = merge([h13_conv_U_a, h13_conv_U_b], mode='max', concat_axis=ch_axis)
    h13_conv_U = BatchNormalization(axis=ch_axis)(h13_conv_U)

    h13_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_V)
    h13_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_V)
    h13_conv_V = merge([h13_conv_V_a, h13_conv_V_b], mode='max', concat_axis=ch_axis)
    h13_conv_V = BatchNormalization(axis=ch_axis)(h13_conv_V)

    h14_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_Y)
    h14_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_Y)
    h14_conv_Y = merge([h14_conv_Y_a, h14_conv_Y_b], mode='max', concat_axis=ch_axis)
    h14_conv_Y = BatchNormalization(axis=ch_axis)(h14_conv_Y)

    h14_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_U)
    h14_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_U)
    h14_conv_U = merge([h14_conv_U_a, h14_conv_U_b], mode='max', concat_axis=ch_axis)
    h14_conv_U = BatchNormalization(axis=ch_axis)(h14_conv_U)

    h14_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_V)
    h14_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_V)
    h14_conv_V = merge([h14_conv_V_a, h14_conv_V_b], mode='max', concat_axis=ch_axis)
    h14_conv_V = BatchNormalization(axis=ch_axis)(h14_conv_V)

    h15_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_Y)
    h15_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_Y)
    h15_conv_Y = merge([h15_conv_Y_a, h15_conv_Y_b], mode='max', concat_axis=ch_axis)
    h15_conv_Y = BatchNormalization(axis=ch_axis)(h15_conv_Y)

    h15_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_U)
    h15_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_U)
    h15_conv_U = merge([h15_conv_U_a, h15_conv_U_b], mode='max', concat_axis=ch_axis)
    h15_conv_U = BatchNormalization(axis=ch_axis)(h15_conv_U)

    h15_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_V)
    h15_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_V)
    h15_conv_V = merge([h15_conv_V_a, h15_conv_V_b], mode='max', concat_axis=ch_axis)
    h15_conv_V = BatchNormalization(axis=ch_axis)(h15_conv_V)

    h16_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_Y)
    h16_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_Y)
    h16_conv_Y = merge([h16_conv_Y_a, h16_conv_Y_b], mode='max', concat_axis=ch_axis)
    h16_conv_Y = BatchNormalization(axis=ch_axis)(h16_conv_Y)

    h16_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_U)
    h16_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_U)
    h16_conv_U = merge([h16_conv_U_a, h16_conv_U_b], mode='max', concat_axis=ch_axis)
    h16_conv_U = BatchNormalization(axis=ch_axis)(h16_conv_U)

    h16_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_V)
    h16_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_V)
    h16_conv_V = merge([h16_conv_V_a, h16_conv_V_b], mode='max', concat_axis=ch_axis)
    h16_conv_V = BatchNormalization(axis=ch_axis)(h16_conv_V)

    poolY = MaxPooling2D(pool_size=(8, 8))(h16_conv_Y)
    poolU = MaxPooling2D(pool_size=(8, 8))(h16_conv_U)
    poolV = MaxPooling2D(pool_size=(8, 8))(h16_conv_V)

    poolY = Dropout(0.2)(poolY)
    poolU = Dropout(0.2)(poolU)
    poolV = Dropout(0.2)(poolV)

    concat_map = merge([poolY, poolU, poolV], mode='concat', concat_axis=ch_axis)

    h16 = Flatten()(concat_map)
    h17 = MaxoutDense(500, nb_feature=5, init='glorot_uniform', W_regularizer=l2(0.0005))(h16)
    h17 = BatchNormalization(axis=1)(h17)
    h17_drop = Dropout(0.2)(h17)
    out = Dense(nb_classes, activation='softmax', init='glorot_uniform', W_regularizer=l2(0.0005))(h17)

    model = Model(input=inputYUV, output=out)
    return model

def get_xfitnet_nincon2(nb_classes, input_shape, ch_axis):
    inputYUV = Input(shape=input_shape)
    inputNorm = BatchNormalization(axis=ch_axis)(inputYUV)

    inp = inputNorm
    if K.image_dim_ordering() == 'tf':
        lb1 = Lambda(lambda x: x[:, :, :, 0:1], output_shape=(32, 32, 1))(inp)
        lb2 = Lambda(lambda x: x[:, :, :, 1:2], output_shape=(32, 32, 1))(inp)
        lb3 = Lambda(lambda x: x[:, :, :, 2:3], output_shape=(32, 32, 1))(inp)
    else:
        lb1 = Lambda(lambda x: x[:, 0:1, :, :], output_shape=(1, 32, 32))(inp)
        lb2 = Lambda(lambda x: x[:, 1:2, :, :], output_shape=(1, 32, 32))(inp)
        lb3 = Lambda(lambda x: x[:, 2:3, :, :], output_shape=(1, 32, 32))(inp)

    inputY = lb1
    inputU = lb2
    inputV = lb3

    inputY_drop = Dropout(0.2)(inputY)
    inputU_drop = Dropout(0.2)(inputU)
    inputV_drop = Dropout(0.2)(inputV)

    h0_conv_Y_a = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputY)
    h0_conv_Y_b = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputY)
    h0_conv_Y = merge([h0_conv_Y_a, h0_conv_Y_b], mode='max', concat_axis=ch_axis)
    h0_conv_Y = BatchNormalization(axis=ch_axis)(h0_conv_Y)

    h0_conv_U_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputU)
    h0_conv_U_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputU)
    h0_conv_U = merge([h0_conv_U_a, h0_conv_U_b], mode='max', concat_axis=ch_axis)
    h0_conv_U = BatchNormalization(axis=ch_axis)(h0_conv_U)

    h0_conv_V_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputV)
    h0_conv_V_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputV)
    h0_conv_V = merge([h0_conv_V_a, h0_conv_V_b], mode='max', concat_axis=ch_axis)
    h0_conv_V = BatchNormalization(axis=ch_axis)(h0_conv_V)

    h1_conv_Y_a = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_Y)
    h1_conv_Y_b = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_Y)
    h1_conv_Y = merge([h1_conv_Y_a, h1_conv_Y_b], mode='max', concat_axis=ch_axis)
    h1_conv_Y = BatchNormalization(axis=ch_axis)(h1_conv_Y)

    h1_conv_U_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_U)
    h1_conv_U_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_U)
    h1_conv_U = merge([h1_conv_U_a, h1_conv_U_b], mode='max', concat_axis=ch_axis)
    h1_conv_U = BatchNormalization(axis=ch_axis)(h1_conv_U)

    h1_conv_V_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_V)
    h1_conv_V_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_V)
    h1_conv_V = merge([h1_conv_V_a, h1_conv_V_b], mode='max', concat_axis=ch_axis)
    h1_conv_V = BatchNormalization(axis=ch_axis)(h1_conv_V)

    h2_conv_Y_a = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_Y)
    h2_conv_Y_b = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_Y)
    h2_conv_Y = merge([h2_conv_Y_a, h2_conv_Y_b], mode='max', concat_axis=ch_axis)
    h2_conv_Y = BatchNormalization(axis=ch_axis)(h2_conv_Y)

    h2_conv_U_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_U)
    h2_conv_U_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_U)
    h2_conv_U = merge([h2_conv_U_a, h2_conv_U_b], mode='max', concat_axis=ch_axis)
    h2_conv_U = BatchNormalization(axis=ch_axis)(h2_conv_U)

    h2_conv_V_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_V)
    h2_conv_V_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_V)
    h2_conv_V = merge([h2_conv_V_a, h2_conv_V_b], mode='max', concat_axis=ch_axis)
    h2_conv_V = BatchNormalization(axis=ch_axis)(h2_conv_V)

    h3_conv_Y_a = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_Y)
    h3_conv_Y_b = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_Y)
    h3_conv_Y = merge([h3_conv_Y_a, h3_conv_Y_b], mode='max', concat_axis=ch_axis)
    h3_conv_Y = BatchNormalization(axis=ch_axis)(h3_conv_Y)

    h3_conv_U_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_U)
    h3_conv_U_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_U)
    h3_conv_U = merge([h3_conv_U_a, h3_conv_U_b], mode='max', concat_axis=ch_axis)
    h3_conv_U = BatchNormalization(axis=ch_axis)(h3_conv_U)

    h3_conv_V_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_V)
    h3_conv_V_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_V)
    h3_conv_V = merge([h3_conv_V_a, h3_conv_V_b], mode='max', concat_axis=ch_axis)
    h3_conv_V = BatchNormalization(axis=ch_axis)(h3_conv_V)

    h4_conv_Y_a = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_Y)
    h4_conv_Y_b = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_Y)
    h4_conv_Y = merge([h4_conv_Y_a, h4_conv_Y_b], mode='max', concat_axis=ch_axis)
    h4_conv_Y = BatchNormalization(axis=ch_axis)(h4_conv_Y)

    h4_conv_U_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_U)
    h4_conv_U_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_U)
    h4_conv_U = merge([h4_conv_U_a, h4_conv_U_b], mode='max', concat_axis=ch_axis)
    h4_conv_U = BatchNormalization(axis=ch_axis)(h4_conv_U)

    h4_conv_V_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_V)
    h4_conv_V_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_V)
    h4_conv_V = merge([h4_conv_V_a, h4_conv_V_b], mode='max', concat_axis=ch_axis)
    h4_conv_V = BatchNormalization(axis=ch_axis)(h4_conv_V)

    poolY = MaxPooling2D(pool_size=(2, 2))(h4_conv_Y)
    poolU = MaxPooling2D(pool_size=(2, 2))(h4_conv_U)
    poolV = MaxPooling2D(pool_size=(2, 2))(h4_conv_V)

    poolY = Dropout(0.2)(poolY)
    poolU = Dropout(0.2)(poolU)
    poolV = Dropout(0.2)(poolV)

    # Inline connections
    Y_to_Y_a = Convolution2D(36, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_Y_b = Convolution2D(36, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_Y = merge([Y_to_Y_a, Y_to_Y_b], mode='max', concat_axis=ch_axis)
    Y_to_Y = BatchNormalization(axis=ch_axis)(Y_to_Y)

    U_to_U_a = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_U_b = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_U = merge([U_to_U_a, U_to_U_b], mode='max', concat_axis=ch_axis)
    U_to_U = BatchNormalization(axis=ch_axis)(U_to_U)

    V_to_V_a = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_V_b = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_V = merge([V_to_V_a, V_to_V_b], mode='max', concat_axis=ch_axis)
    V_to_V = BatchNormalization(axis=ch_axis)(Y_to_Y)

    # Cross connections: Y <-> U, Y <-> V
    Y_to_UV_a = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_UV_b = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_UV = merge([Y_to_UV_a, Y_to_UV_b], mode='max', concat_axis=ch_axis)
    Y_to_UV = BatchNormalization(axis=ch_axis)(Y_to_UV)

    U_to_Y_a = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_Y_b = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_Y = merge([U_to_Y_a, U_to_Y_b], mode='max', concat_axis=ch_axis)
    U_to_Y = BatchNormalization(axis=ch_axis)(U_to_Y)

    V_to_Y_a = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_Y_b = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_Y = merge([V_to_Y_a, V_to_Y_b], mode='max', concat_axis=ch_axis)
    V_to_Y = BatchNormalization(axis=ch_axis)(V_to_Y)

    Ymap = merge([Y_to_Y, U_to_Y, V_to_Y], mode='concat', concat_axis=ch_axis)
    Umap = merge([U_to_U, Y_to_UV], mode='concat', concat_axis=ch_axis)
    Vmap = merge([V_to_V, Y_to_UV], mode='concat', concat_axis=ch_axis)

    h5_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h5_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h5_conv_Y = merge([h5_conv_Y_a, h5_conv_Y_b], mode='max', concat_axis=ch_axis)
    h5_conv_Y = BatchNormalization(axis=ch_axis)(h5_conv_Y)

    h5_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h5_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h5_conv_U = merge([h5_conv_U_a, h5_conv_U_b], mode='max', concat_axis=ch_axis)
    h5_conv_U = BatchNormalization(axis=ch_axis)(h5_conv_U)

    h5_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h5_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h5_conv_V = merge([h5_conv_V_a, h5_conv_V_b], mode='max', concat_axis=ch_axis)
    h5_conv_V = BatchNormalization(axis=ch_axis)(h5_conv_V)

    h6_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_Y)
    h6_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_Y)
    h6_conv_Y = merge([h6_conv_Y_a, h6_conv_Y_b], mode='max', concat_axis=ch_axis)
    h6_conv_Y = BatchNormalization(axis=ch_axis)(h6_conv_Y)

    h6_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_U)
    h6_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_U)
    h6_conv_U = merge([h6_conv_U_a, h6_conv_U_b], mode='max', concat_axis=ch_axis)
    h6_conv_U = BatchNormalization(axis=ch_axis)(h6_conv_U)

    h6_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_V)
    h6_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_V)
    h6_conv_V = merge([h6_conv_V_a, h6_conv_V_b], mode='max', concat_axis=ch_axis)
    h6_conv_V = BatchNormalization(axis=ch_axis)(h6_conv_V)

    h7_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_Y)
    h7_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_Y)
    h7_conv_Y = merge([h7_conv_Y_a, h7_conv_Y_b], mode='max', concat_axis=ch_axis)
    h7_conv_Y = BatchNormalization(axis=ch_axis)(h7_conv_Y)

    h7_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_U)
    h7_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_U)
    h7_conv_U = merge([h7_conv_U_a, h7_conv_U_b], mode='max', concat_axis=ch_axis)
    h7_conv_U = BatchNormalization(axis=ch_axis)(h7_conv_U)

    h7_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_V)
    h7_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_V)
    h7_conv_V = merge([h7_conv_V_a, h7_conv_V_b], mode='max', concat_axis=ch_axis)
    h7_conv_V = BatchNormalization(axis=ch_axis)(h7_conv_V)

    h8_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_Y)
    h8_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_Y)
    h8_conv_Y = merge([h8_conv_Y_a, h8_conv_Y_b], mode='max', concat_axis=ch_axis)
    h8_conv_Y = BatchNormalization(axis=ch_axis)(h8_conv_Y)

    h8_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_U)
    h8_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_U)
    h8_conv_U = merge([h8_conv_U_a, h8_conv_U_b], mode='max', concat_axis=ch_axis)
    h8_conv_U = BatchNormalization(axis=ch_axis)(h8_conv_U)

    h8_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_V)
    h8_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_V)
    h8_conv_V = merge([h8_conv_V_a, h8_conv_V_b], mode='max', concat_axis=ch_axis)
    h8_conv_V = BatchNormalization(axis=ch_axis)(h8_conv_V)

    h9_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_Y)
    h9_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_Y)
    h9_conv_Y = merge([h9_conv_Y_a, h9_conv_Y_b], mode='max', concat_axis=ch_axis)
    h9_conv_Y = BatchNormalization(axis=ch_axis)(h9_conv_Y)

    h9_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_U)
    h9_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_U)
    h9_conv_U = merge([h9_conv_U_a, h9_conv_U_b], mode='max', concat_axis=ch_axis)
    h9_conv_U = BatchNormalization(axis=ch_axis)(h9_conv_U)

    h9_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_V)
    h9_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_V)
    h9_conv_V = merge([h9_conv_V_a, h9_conv_V_b], mode='max', concat_axis=ch_axis)
    h9_conv_V = BatchNormalization(axis=ch_axis)(h9_conv_V)

    h10_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_Y)
    h10_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_Y)
    h10_conv_Y = merge([h10_conv_Y_a, h10_conv_Y_b], mode='max', concat_axis=ch_axis)
    h10_conv_Y = BatchNormalization(axis=ch_axis)(h10_conv_Y)

    h10_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_U)
    h10_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_U)
    h10_conv_U = merge([h10_conv_U_a, h10_conv_U_b], mode='max', concat_axis=ch_axis)
    h10_conv_U = BatchNormalization(axis=ch_axis)(h10_conv_U)

    h10_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_V)
    h10_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_V)
    h10_conv_V = merge([h10_conv_V_a, h10_conv_V_b], mode='max', concat_axis=ch_axis)
    h10_conv_V = BatchNormalization(axis=ch_axis)(h10_conv_V)

    poolY = MaxPooling2D(pool_size=(2, 2))(h10_conv_Y)
    poolU = MaxPooling2D(pool_size=(2, 2))(h10_conv_U)
    poolV = MaxPooling2D(pool_size=(2, 2))(h10_conv_V)

    poolY = Dropout(0.2)(poolY)
    poolU = Dropout(0.2)(poolU)
    poolV = Dropout(0.2)(poolV)

    U_to_Y = Convolution2D(30, 1, 1, border_mode='same', activation='relu')(poolU)
    U_to_Y = Convolution2D(30, 1, 1, border_mode='same', activation='relu')(U_to_Y)
    U_to_V = Convolution2D(30, 1, 1, border_mode='same', activation='relu')(poolU)
    U_to_V = Convolution2D(30, 1, 1, border_mode='same', activation='relu')(U_to_V)

    V_to_Y = Convolution2D(30, 1, 1, border_mode='same', activation='relu')(poolV)
    V_to_Y = Convolution2D(30, 1, 1, border_mode='same', activation='relu')(V_to_Y)
    V_to_U = Convolution2D(30, 1, 1, border_mode='same', activation='relu')(poolV)
    V_to_U = Convolution2D(30, 1, 1, border_mode='same', activation='relu')(V_to_U)

    Y_to_U = Convolution2D(60, 1, 1, border_mode='same', activation='relu')(poolY)
    Y_to_U = Convolution2D(60, 1, 1, border_mode='same', activation='relu')(Y_to_U)
    Y_to_V = Convolution2D(60, 1, 1, border_mode='same', activation='relu')(poolY)
    Y_to_V = Convolution2D(60, 1, 1, border_mode='same', activation='relu')(Y_to_V)

    Ymap = merge([poolY, U_to_Y, V_to_Y], mode='concat', concat_axis=ch_axis)
    Umap = merge([poolU, Y_to_U, V_to_U], mode='concat', concat_axis=ch_axis)
    Vmap = merge([poolV, Y_to_V, U_to_V], mode='concat', concat_axis=ch_axis)

    h11_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h11_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h11_conv_Y = merge([h11_conv_Y_a, h11_conv_Y_b], mode='max', concat_axis=ch_axis)
    h11_conv_Y = BatchNormalization(axis=ch_axis)(h11_conv_Y)

    h11_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h11_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h11_conv_U = merge([h11_conv_U_a, h11_conv_U_b], mode='max', concat_axis=ch_axis)
    h11_conv_U = BatchNormalization(axis=ch_axis)(h11_conv_U)

    h11_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h11_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h11_conv_V = merge([h11_conv_V_a, h11_conv_V_b], mode='max', concat_axis=ch_axis)
    h11_conv_V = BatchNormalization(axis=ch_axis)(h11_conv_V)

    h12_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_Y)
    h12_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_Y)
    h12_conv_Y = merge([h12_conv_Y_a, h12_conv_Y_b], mode='max', concat_axis=ch_axis)
    h12_conv_Y = BatchNormalization(axis=ch_axis)(h12_conv_Y)

    h12_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_U)
    h12_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_U)
    h12_conv_U = merge([h12_conv_U_a, h12_conv_U_b], mode='max', concat_axis=ch_axis)
    h12_conv_U = BatchNormalization(axis=ch_axis)(h12_conv_U)

    h12_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_V)
    h12_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_V)
    h12_conv_V = merge([h12_conv_V_a, h12_conv_V_b], mode='max', concat_axis=ch_axis)
    h12_conv_V = BatchNormalization(axis=ch_axis)(h12_conv_V)

    h13_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_Y)
    h13_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_Y)
    h13_conv_Y = merge([h13_conv_Y_a, h13_conv_Y_b], mode='max', concat_axis=ch_axis)
    h13_conv_Y = BatchNormalization(axis=ch_axis)(h13_conv_Y)

    h13_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_U)
    h13_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_U)
    h13_conv_U = merge([h13_conv_U_a, h13_conv_U_b], mode='max', concat_axis=ch_axis)
    h13_conv_U = BatchNormalization(axis=ch_axis)(h13_conv_U)

    h13_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_V)
    h13_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_V)
    h13_conv_V = merge([h13_conv_V_a, h13_conv_V_b], mode='max', concat_axis=ch_axis)
    h13_conv_V = BatchNormalization(axis=ch_axis)(h13_conv_V)

    h14_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_Y)
    h14_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_Y)
    h14_conv_Y = merge([h14_conv_Y_a, h14_conv_Y_b], mode='max', concat_axis=ch_axis)
    h14_conv_Y = BatchNormalization(axis=ch_axis)(h14_conv_Y)

    h14_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_U)
    h14_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_U)
    h14_conv_U = merge([h14_conv_U_a, h14_conv_U_b], mode='max', concat_axis=ch_axis)
    h14_conv_U = BatchNormalization(axis=ch_axis)(h14_conv_U)

    h14_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_V)
    h14_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_V)
    h14_conv_V = merge([h14_conv_V_a, h14_conv_V_b], mode='max', concat_axis=ch_axis)
    h14_conv_V = BatchNormalization(axis=ch_axis)(h14_conv_V)

    h15_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_Y)
    h15_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_Y)
    h15_conv_Y = merge([h15_conv_Y_a, h15_conv_Y_b], mode='max', concat_axis=ch_axis)
    h15_conv_Y = BatchNormalization(axis=ch_axis)(h15_conv_Y)

    h15_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_U)
    h15_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_U)
    h15_conv_U = merge([h15_conv_U_a, h15_conv_U_b], mode='max', concat_axis=ch_axis)
    h15_conv_U = BatchNormalization(axis=ch_axis)(h15_conv_U)

    h15_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_V)
    h15_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_V)
    h15_conv_V = merge([h15_conv_V_a, h15_conv_V_b], mode='max', concat_axis=ch_axis)
    h15_conv_V = BatchNormalization(axis=ch_axis)(h15_conv_V)

    h16_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_Y)
    h16_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_Y)
    h16_conv_Y = merge([h16_conv_Y_a, h16_conv_Y_b], mode='max', concat_axis=ch_axis)
    h16_conv_Y = BatchNormalization(axis=ch_axis)(h16_conv_Y)

    h16_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_U)
    h16_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_U)
    h16_conv_U = merge([h16_conv_U_a, h16_conv_U_b], mode='max', concat_axis=ch_axis)
    h16_conv_U = BatchNormalization(axis=ch_axis)(h16_conv_U)

    h16_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_V)
    h16_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_V)
    h16_conv_V = merge([h16_conv_V_a, h16_conv_V_b], mode='max', concat_axis=ch_axis)
    h16_conv_V = BatchNormalization(axis=ch_axis)(h16_conv_V)

    poolY = MaxPooling2D(pool_size=(8, 8))(h16_conv_Y)
    poolU = MaxPooling2D(pool_size=(8, 8))(h16_conv_U)
    poolV = MaxPooling2D(pool_size=(8, 8))(h16_conv_V)

    poolY = Dropout(0.2)(poolY)
    poolU = Dropout(0.2)(poolU)
    poolV = Dropout(0.2)(poolV)

    concat_map = merge([poolY, poolU, poolV], mode='concat', concat_axis=ch_axis)

    h16 = Flatten()(concat_map)
    h17 = MaxoutDense(500, nb_feature=5, init='glorot_uniform', W_regularizer=l2(0.0005))(h16)
    h17 = BatchNormalization(axis=1)(h17)
    h17_drop = Dropout(0.2)(h17)
    out = Dense(nb_classes, activation='softmax', init='glorot_uniform', W_regularizer=l2(0.0005))(h17)

    model = Model(input=inputYUV, output=out)
    return model

def get_xfitnet_nincon2elu(nb_classes, input_shape, ch_axis):
    inputYUV = Input(shape=input_shape)
    inputNorm = BatchNormalization(axis=ch_axis)(inputYUV)

    inp = inputNorm
    if K.image_dim_ordering() == 'tf':
        lb1 = Lambda(lambda x: x[:, :, :, 0:1], output_shape=(32, 32, 1))(inp)
        lb2 = Lambda(lambda x: x[:, :, :, 1:2], output_shape=(32, 32, 1))(inp)
        lb3 = Lambda(lambda x: x[:, :, :, 2:3], output_shape=(32, 32, 1))(inp)
    else:
        lb1 = Lambda(lambda x: x[:, 0:1, :, :], output_shape=(1, 32, 32))(inp)
        lb2 = Lambda(lambda x: x[:, 1:2, :, :], output_shape=(1, 32, 32))(inp)
        lb3 = Lambda(lambda x: x[:, 2:3, :, :], output_shape=(1, 32, 32))(inp)

    inputY = lb1
    inputU = lb2
    inputV = lb3

    inputY_drop = Dropout(0.2)(inputY)
    inputU_drop = Dropout(0.2)(inputU)
    inputV_drop = Dropout(0.2)(inputV)

    h0_conv_Y_a = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputY)
    h0_conv_Y_b = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputY)
    h0_conv_Y = merge([h0_conv_Y_a, h0_conv_Y_b], mode='max', concat_axis=ch_axis)
    h0_conv_Y = BatchNormalization(axis=ch_axis)(h0_conv_Y)

    h0_conv_U_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputU)
    h0_conv_U_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputU)
    h0_conv_U = merge([h0_conv_U_a, h0_conv_U_b], mode='max', concat_axis=ch_axis)
    h0_conv_U = BatchNormalization(axis=ch_axis)(h0_conv_U)

    h0_conv_V_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputV)
    h0_conv_V_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputV)
    h0_conv_V = merge([h0_conv_V_a, h0_conv_V_b], mode='max', concat_axis=ch_axis)
    h0_conv_V = BatchNormalization(axis=ch_axis)(h0_conv_V)

    h1_conv_Y_a = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_Y)
    h1_conv_Y_b = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_Y)
    h1_conv_Y = merge([h1_conv_Y_a, h1_conv_Y_b], mode='max', concat_axis=ch_axis)
    h1_conv_Y = BatchNormalization(axis=ch_axis)(h1_conv_Y)

    h1_conv_U_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_U)
    h1_conv_U_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_U)
    h1_conv_U = merge([h1_conv_U_a, h1_conv_U_b], mode='max', concat_axis=ch_axis)
    h1_conv_U = BatchNormalization(axis=ch_axis)(h1_conv_U)

    h1_conv_V_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_V)
    h1_conv_V_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h0_conv_V)
    h1_conv_V = merge([h1_conv_V_a, h1_conv_V_b], mode='max', concat_axis=ch_axis)
    h1_conv_V = BatchNormalization(axis=ch_axis)(h1_conv_V)

    h2_conv_Y_a = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_Y)
    h2_conv_Y_b = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_Y)
    h2_conv_Y = merge([h2_conv_Y_a, h2_conv_Y_b], mode='max', concat_axis=ch_axis)
    h2_conv_Y = BatchNormalization(axis=ch_axis)(h2_conv_Y)

    h2_conv_U_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_U)
    h2_conv_U_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_U)
    h2_conv_U = merge([h2_conv_U_a, h2_conv_U_b], mode='max', concat_axis=ch_axis)
    h2_conv_U = BatchNormalization(axis=ch_axis)(h2_conv_U)

    h2_conv_V_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_V)
    h2_conv_V_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h1_conv_V)
    h2_conv_V = merge([h2_conv_V_a, h2_conv_V_b], mode='max', concat_axis=ch_axis)
    h2_conv_V = BatchNormalization(axis=ch_axis)(h2_conv_V)

    h3_conv_Y_a = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_Y)
    h3_conv_Y_b = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_Y)
    h3_conv_Y = merge([h3_conv_Y_a, h3_conv_Y_b], mode='max', concat_axis=ch_axis)
    h3_conv_Y = BatchNormalization(axis=ch_axis)(h3_conv_Y)

    h3_conv_U_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_U)
    h3_conv_U_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_U)
    h3_conv_U = merge([h3_conv_U_a, h3_conv_U_b], mode='max', concat_axis=ch_axis)
    h3_conv_U = BatchNormalization(axis=ch_axis)(h3_conv_U)

    h3_conv_V_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_V)
    h3_conv_V_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h2_conv_V)
    h3_conv_V = merge([h3_conv_V_a, h3_conv_V_b], mode='max', concat_axis=ch_axis)
    h3_conv_V = BatchNormalization(axis=ch_axis)(h3_conv_V)

    h4_conv_Y_a = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_Y)
    h4_conv_Y_b = Convolution2D(36, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_Y)
    h4_conv_Y = merge([h4_conv_Y_a, h4_conv_Y_b], mode='max', concat_axis=ch_axis)
    h4_conv_Y = BatchNormalization(axis=ch_axis)(h4_conv_Y)

    h4_conv_U_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_U)
    h4_conv_U_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_U)
    h4_conv_U = merge([h4_conv_U_a, h4_conv_U_b], mode='max', concat_axis=ch_axis)
    h4_conv_U = BatchNormalization(axis=ch_axis)(h4_conv_U)

    h4_conv_V_a = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_V)
    h4_conv_V_b = Convolution2D(18, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h3_conv_V)
    h4_conv_V = merge([h4_conv_V_a, h4_conv_V_b], mode='max', concat_axis=ch_axis)
    h4_conv_V = BatchNormalization(axis=ch_axis)(h4_conv_V)

    poolY = MaxPooling2D(pool_size=(2, 2))(h4_conv_Y)
    poolU = MaxPooling2D(pool_size=(2, 2))(h4_conv_U)
    poolV = MaxPooling2D(pool_size=(2, 2))(h4_conv_V)

    poolY = Dropout(0.2)(poolY)
    poolU = Dropout(0.2)(poolU)
    poolV = Dropout(0.2)(poolV)

    # Inline connections
    Y_to_Y_a = Convolution2D(36, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_Y_b = Convolution2D(36, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_Y = merge([Y_to_Y_a, Y_to_Y_b], mode='max', concat_axis=ch_axis)
    Y_to_Y = BatchNormalization(axis=ch_axis)(Y_to_Y)

    U_to_U_a = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_U_b = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_U = merge([U_to_U_a, U_to_U_b], mode='max', concat_axis=ch_axis)
    U_to_U = BatchNormalization(axis=ch_axis)(U_to_U)

    V_to_V_a = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_V_b = Convolution2D(18, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_V = merge([V_to_V_a, V_to_V_b], mode='max', concat_axis=ch_axis)
    V_to_V = BatchNormalization(axis=ch_axis)(Y_to_Y)

    # Cross connections: Y <-> U, Y <-> V
    Y_to_UV_a = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_UV_b = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
    Y_to_UV = merge([Y_to_UV_a, Y_to_UV_b], mode='max', concat_axis=ch_axis)
    Y_to_UV = BatchNormalization(axis=ch_axis)(Y_to_UV)

    U_to_Y_a = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_Y_b = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
    U_to_Y = merge([U_to_Y_a, U_to_Y_b], mode='max', concat_axis=ch_axis)
    U_to_Y = BatchNormalization(axis=ch_axis)(U_to_Y)

    V_to_Y_a = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_Y_b = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
    V_to_Y = merge([V_to_Y_a, V_to_Y_b], mode='max', concat_axis=ch_axis)
    V_to_Y = BatchNormalization(axis=ch_axis)(V_to_Y)

    Ymap = merge([Y_to_Y, U_to_Y, V_to_Y], mode='concat', concat_axis=ch_axis)
    Umap = merge([U_to_U, Y_to_UV], mode='concat', concat_axis=ch_axis)
    Vmap = merge([V_to_V, Y_to_UV], mode='concat', concat_axis=ch_axis)

    h5_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h5_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h5_conv_Y = merge([h5_conv_Y_a, h5_conv_Y_b], mode='max', concat_axis=ch_axis)
    h5_conv_Y = BatchNormalization(axis=ch_axis)(h5_conv_Y)

    h5_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h5_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h5_conv_U = merge([h5_conv_U_a, h5_conv_U_b], mode='max', concat_axis=ch_axis)
    h5_conv_U = BatchNormalization(axis=ch_axis)(h5_conv_U)

    h5_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h5_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h5_conv_V = merge([h5_conv_V_a, h5_conv_V_b], mode='max', concat_axis=ch_axis)
    h5_conv_V = BatchNormalization(axis=ch_axis)(h5_conv_V)

    h6_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_Y)
    h6_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_Y)
    h6_conv_Y = merge([h6_conv_Y_a, h6_conv_Y_b], mode='max', concat_axis=ch_axis)
    h6_conv_Y = BatchNormalization(axis=ch_axis)(h6_conv_Y)

    h6_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_U)
    h6_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_U)
    h6_conv_U = merge([h6_conv_U_a, h6_conv_U_b], mode='max', concat_axis=ch_axis)
    h6_conv_U = BatchNormalization(axis=ch_axis)(h6_conv_U)

    h6_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_V)
    h6_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h5_conv_V)
    h6_conv_V = merge([h6_conv_V_a, h6_conv_V_b], mode='max', concat_axis=ch_axis)
    h6_conv_V = BatchNormalization(axis=ch_axis)(h6_conv_V)

    h7_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_Y)
    h7_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_Y)
    h7_conv_Y = merge([h7_conv_Y_a, h7_conv_Y_b], mode='max', concat_axis=ch_axis)
    h7_conv_Y = BatchNormalization(axis=ch_axis)(h7_conv_Y)

    h7_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_U)
    h7_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_U)
    h7_conv_U = merge([h7_conv_U_a, h7_conv_U_b], mode='max', concat_axis=ch_axis)
    h7_conv_U = BatchNormalization(axis=ch_axis)(h7_conv_U)

    h7_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_V)
    h7_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h6_conv_V)
    h7_conv_V = merge([h7_conv_V_a, h7_conv_V_b], mode='max', concat_axis=ch_axis)
    h7_conv_V = BatchNormalization(axis=ch_axis)(h7_conv_V)

    h8_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_Y)
    h8_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_Y)
    h8_conv_Y = merge([h8_conv_Y_a, h8_conv_Y_b], mode='max', concat_axis=ch_axis)
    h8_conv_Y = BatchNormalization(axis=ch_axis)(h8_conv_Y)

    h8_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_U)
    h8_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_U)
    h8_conv_U = merge([h8_conv_U_a, h8_conv_U_b], mode='max', concat_axis=ch_axis)
    h8_conv_U = BatchNormalization(axis=ch_axis)(h8_conv_U)

    h8_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_V)
    h8_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h7_conv_V)
    h8_conv_V = merge([h8_conv_V_a, h8_conv_V_b], mode='max', concat_axis=ch_axis)
    h8_conv_V = BatchNormalization(axis=ch_axis)(h8_conv_V)

    h9_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_Y)
    h9_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_Y)
    h9_conv_Y = merge([h9_conv_Y_a, h9_conv_Y_b], mode='max', concat_axis=ch_axis)
    h9_conv_Y = BatchNormalization(axis=ch_axis)(h9_conv_Y)

    h9_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_U)
    h9_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_U)
    h9_conv_U = merge([h9_conv_U_a, h9_conv_U_b], mode='max', concat_axis=ch_axis)
    h9_conv_U = BatchNormalization(axis=ch_axis)(h9_conv_U)

    h9_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_V)
    h9_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h8_conv_V)
    h9_conv_V = merge([h9_conv_V_a, h9_conv_V_b], mode='max', concat_axis=ch_axis)
    h9_conv_V = BatchNormalization(axis=ch_axis)(h9_conv_V)

    h10_conv_Y_a = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_Y)
    h10_conv_Y_b = Convolution2D(60, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_Y)
    h10_conv_Y = merge([h10_conv_Y_a, h10_conv_Y_b], mode='max', concat_axis=ch_axis)
    h10_conv_Y = BatchNormalization(axis=ch_axis)(h10_conv_Y)

    h10_conv_U_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_U)
    h10_conv_U_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_U)
    h10_conv_U = merge([h10_conv_U_a, h10_conv_U_b], mode='max', concat_axis=ch_axis)
    h10_conv_U = BatchNormalization(axis=ch_axis)(h10_conv_U)

    h10_conv_V_a = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_V)
    h10_conv_V_b = Convolution2D(30, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h9_conv_V)
    h10_conv_V = merge([h10_conv_V_a, h10_conv_V_b], mode='max', concat_axis=ch_axis)
    h10_conv_V = BatchNormalization(axis=ch_axis)(h10_conv_V)

    poolY = MaxPooling2D(pool_size=(2, 2))(h10_conv_Y)
    poolU = MaxPooling2D(pool_size=(2, 2))(h10_conv_U)
    poolV = MaxPooling2D(pool_size=(2, 2))(h10_conv_V)

    poolY = Dropout(0.2)(poolY)
    poolU = Dropout(0.2)(poolU)
    poolV = Dropout(0.2)(poolV)

    U_to_Y = Convolution2D(30, 1, 1, border_mode='same', activation='elu')(poolU)
    U_to_Y = Convolution2D(30, 1, 1, border_mode='same', activation='elu')(U_to_Y)
    U_to_V = Convolution2D(30, 1, 1, border_mode='same', activation='elu')(poolU)
    U_to_V = Convolution2D(30, 1, 1, border_mode='same', activation='elu')(U_to_V)

    V_to_Y = Convolution2D(30, 1, 1, border_mode='same', activation='elu')(poolV)
    V_to_Y = Convolution2D(30, 1, 1, border_mode='same', activation='elu')(V_to_Y)
    V_to_U = Convolution2D(30, 1, 1, border_mode='same', activation='elu')(poolV)
    V_to_U = Convolution2D(30, 1, 1, border_mode='same', activation='elu')(V_to_U)

    Y_to_U = Convolution2D(60, 1, 1, border_mode='same', activation='elu')(poolY)
    Y_to_U = Convolution2D(60, 1, 1, border_mode='same', activation='elu')(Y_to_U)
    Y_to_V = Convolution2D(60, 1, 1, border_mode='same', activation='elu')(poolY)
    Y_to_V = Convolution2D(60, 1, 1, border_mode='same', activation='elu')(Y_to_V)

    Ymap = merge([poolY, U_to_Y, V_to_Y], mode='concat', concat_axis=ch_axis)
    Umap = merge([poolU, Y_to_U, V_to_U], mode='concat', concat_axis=ch_axis)
    Vmap = merge([poolV, Y_to_V, U_to_V], mode='concat', concat_axis=ch_axis)

    h11_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h11_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
    h11_conv_Y = merge([h11_conv_Y_a, h11_conv_Y_b], mode='max', concat_axis=ch_axis)
    h11_conv_Y = BatchNormalization(axis=ch_axis)(h11_conv_Y)

    h11_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h11_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
    h11_conv_U = merge([h11_conv_U_a, h11_conv_U_b], mode='max', concat_axis=ch_axis)
    h11_conv_U = BatchNormalization(axis=ch_axis)(h11_conv_U)

    h11_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h11_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
    h11_conv_V = merge([h11_conv_V_a, h11_conv_V_b], mode='max', concat_axis=ch_axis)
    h11_conv_V = BatchNormalization(axis=ch_axis)(h11_conv_V)

    h12_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_Y)
    h12_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_Y)
    h12_conv_Y = merge([h12_conv_Y_a, h12_conv_Y_b], mode='max', concat_axis=ch_axis)
    h12_conv_Y = BatchNormalization(axis=ch_axis)(h12_conv_Y)

    h12_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_U)
    h12_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_U)
    h12_conv_U = merge([h12_conv_U_a, h12_conv_U_b], mode='max', concat_axis=ch_axis)
    h12_conv_U = BatchNormalization(axis=ch_axis)(h12_conv_U)

    h12_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_V)
    h12_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h11_conv_V)
    h12_conv_V = merge([h12_conv_V_a, h12_conv_V_b], mode='max', concat_axis=ch_axis)
    h12_conv_V = BatchNormalization(axis=ch_axis)(h12_conv_V)

    h13_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_Y)
    h13_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_Y)
    h13_conv_Y = merge([h13_conv_Y_a, h13_conv_Y_b], mode='max', concat_axis=ch_axis)
    h13_conv_Y = BatchNormalization(axis=ch_axis)(h13_conv_Y)

    h13_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_U)
    h13_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_U)
    h13_conv_U = merge([h13_conv_U_a, h13_conv_U_b], mode='max', concat_axis=ch_axis)
    h13_conv_U = BatchNormalization(axis=ch_axis)(h13_conv_U)

    h13_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_V)
    h13_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h12_conv_V)
    h13_conv_V = merge([h13_conv_V_a, h13_conv_V_b], mode='max', concat_axis=ch_axis)
    h13_conv_V = BatchNormalization(axis=ch_axis)(h13_conv_V)

    h14_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_Y)
    h14_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_Y)
    h14_conv_Y = merge([h14_conv_Y_a, h14_conv_Y_b], mode='max', concat_axis=ch_axis)
    h14_conv_Y = BatchNormalization(axis=ch_axis)(h14_conv_Y)

    h14_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_U)
    h14_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_U)
    h14_conv_U = merge([h14_conv_U_a, h14_conv_U_b], mode='max', concat_axis=ch_axis)
    h14_conv_U = BatchNormalization(axis=ch_axis)(h14_conv_U)

    h14_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_V)
    h14_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h13_conv_V)
    h14_conv_V = merge([h14_conv_V_a, h14_conv_V_b], mode='max', concat_axis=ch_axis)
    h14_conv_V = BatchNormalization(axis=ch_axis)(h14_conv_V)

    h15_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_Y)
    h15_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_Y)
    h15_conv_Y = merge([h15_conv_Y_a, h15_conv_Y_b], mode='max', concat_axis=ch_axis)
    h15_conv_Y = BatchNormalization(axis=ch_axis)(h15_conv_Y)

    h15_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_U)
    h15_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_U)
    h15_conv_U = merge([h15_conv_U_a, h15_conv_U_b], mode='max', concat_axis=ch_axis)
    h15_conv_U = BatchNormalization(axis=ch_axis)(h15_conv_U)

    h15_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_V)
    h15_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h14_conv_V)
    h15_conv_V = merge([h15_conv_V_a, h15_conv_V_b], mode='max', concat_axis=ch_axis)
    h15_conv_V = BatchNormalization(axis=ch_axis)(h15_conv_V)

    h16_conv_Y_a = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_Y)
    h16_conv_Y_b = Convolution2D(96, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_Y)
    h16_conv_Y = merge([h16_conv_Y_a, h16_conv_Y_b], mode='max', concat_axis=ch_axis)
    h16_conv_Y = BatchNormalization(axis=ch_axis)(h16_conv_Y)

    h16_conv_U_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_U)
    h16_conv_U_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_U)
    h16_conv_U = merge([h16_conv_U_a, h16_conv_U_b], mode='max', concat_axis=ch_axis)
    h16_conv_U = BatchNormalization(axis=ch_axis)(h16_conv_U)

    h16_conv_V_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_V)
    h16_conv_V_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(
        h15_conv_V)
    h16_conv_V = merge([h16_conv_V_a, h16_conv_V_b], mode='max', concat_axis=ch_axis)
    h16_conv_V = BatchNormalization(axis=ch_axis)(h16_conv_V)

    poolY = MaxPooling2D(pool_size=(8, 8))(h16_conv_Y)
    poolU = MaxPooling2D(pool_size=(8, 8))(h16_conv_U)
    poolV = MaxPooling2D(pool_size=(8, 8))(h16_conv_V)

    poolY = Dropout(0.2)(poolY)
    poolU = Dropout(0.2)(poolU)
    poolV = Dropout(0.2)(poolV)

    concat_map = merge([poolY, poolU, poolV], mode='concat', concat_axis=ch_axis)

    h16 = Flatten()(concat_map)
    h17 = MaxoutDense(500, nb_feature=5, init='glorot_uniform', W_regularizer=l2(0.0005))(h16)
    h17 = BatchNormalization(axis=1)(h17)
    h17_drop = Dropout(0.2)(h17)
    out = Dense(nb_classes, activation='softmax', init='glorot_uniform', W_regularizer=l2(0.0005))(h17)

    model = Model(input=inputYUV, output=out)
    return model