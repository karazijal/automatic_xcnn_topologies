from keras.models import Model
from keras.layers import Dense, Dropout, MaxoutDense, Flatten, merge, Input
from keras.regularizers import l2
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

import keras.backend as K
input_shape = (3, 32, 32)
pr_axis = 1
if K.image_dim_ordering() == 'tf':
    input_shape = (32, 32, 3)
    pr_axis = 3

def get_baseline(nb_classes):
    batch_size = 128
    # nb_classes = 10

    inputYUV = Input(shape=input_shape)
    inputNorm = BatchNormalization(axis=pr_axis)(inputYUV)

    input_drop = Dropout(0.2)(inputNorm)

    # This is a single convolutional maxout layer.
    h0_conv_a = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(input_drop)
    h0_conv_b = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(input_drop)
    h0_conv = merge([h0_conv_a, h0_conv_b], mode='max', concat_axis=pr_axis)
    h0_conv = BatchNormalization(axis=pr_axis)(h0_conv)

    h1_conv_a = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h0_conv)
    h1_conv_b = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h0_conv)
    h1_conv = merge([h1_conv_a, h1_conv_b], mode='max', concat_axis=pr_axis)
    h1_conv = BatchNormalization(axis=pr_axis)(h1_conv)

    h2_conv_a = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h1_conv)
    h2_conv_b = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h1_conv)
    h2_conv = merge([h2_conv_a, h2_conv_b], mode='max', concat_axis=pr_axis)
    h2_conv = BatchNormalization(axis=pr_axis)(h2_conv)

    h3_conv_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h2_conv)
    h3_conv_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h2_conv)
    h3_conv = merge([h3_conv_a, h3_conv_b], mode='max', concat_axis=pr_axis)
    h3_conv = BatchNormalization(axis=pr_axis)(h3_conv)

    h4_conv_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h3_conv)
    h4_conv_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h3_conv)
    h4_conv = merge([h4_conv_a, h4_conv_b], mode='max', concat_axis=pr_axis)
    h4_conv = BatchNormalization(axis=pr_axis)(h4_conv)

    h4_pool = MaxPooling2D(pool_size=(2, 2))(h4_conv)
    h4_drop = Dropout(0.2)(h4_pool)

    h5_conv_a = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h4_drop)
    h5_conv_b = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h4_drop)
    h5_conv = merge([h5_conv_a, h5_conv_b], mode='max', concat_axis=pr_axis)
    h5_conv = BatchNormalization(axis=pr_axis)(h5_conv)

    h6_conv_a = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h5_conv)
    h6_conv_b = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h5_conv)
    h6_conv = merge([h6_conv_a, h6_conv_b], mode='max', concat_axis=pr_axis)
    h6_conv = BatchNormalization(axis=pr_axis)(h6_conv)

    h7_conv_a = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h6_conv)
    h7_conv_b = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h6_conv)
    h7_conv = merge([h7_conv_a, h7_conv_b], mode='max', concat_axis=pr_axis)
    h7_conv = BatchNormalization(axis=pr_axis)(h7_conv)

    h8_conv_a = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h7_conv)
    h8_conv_b = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h7_conv)
    h8_conv = merge([h8_conv_a, h8_conv_b], mode='max', concat_axis=pr_axis)
    h8_conv = BatchNormalization(axis=pr_axis)(h8_conv)

    h9_conv_a = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h8_conv)
    h9_conv_b = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h8_conv)
    h9_conv = merge([h9_conv_a, h9_conv_b], mode='max', concat_axis=pr_axis)
    h9_conv = BatchNormalization(axis=pr_axis)(h9_conv)

    h10_conv_a = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h9_conv)
    h10_conv_b = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h9_conv)
    h10_conv = merge([h10_conv_a, h10_conv_b], mode='max', concat_axis=pr_axis)
    h10_conv = BatchNormalization(axis=pr_axis)(h10_conv)

    h10_pool = MaxPooling2D(pool_size=(2, 2))(h10_conv)
    h10_drop = Dropout(0.2)(h10_pool)

    h11_conv_a = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h10_drop)
    h11_conv_b = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h10_drop)
    h11_conv = merge([h11_conv_a, h11_conv_b], mode='max', concat_axis=pr_axis)
    h11_conv = BatchNormalization(axis=pr_axis)(h11_conv)

    h12_conv_a = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h11_conv)
    h12_conv_b = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h11_conv)
    h12_conv = merge([h12_conv_a, h12_conv_b], mode='max', concat_axis=pr_axis)
    h12_conv = BatchNormalization(axis=pr_axis)(h12_conv)

    h13_conv_a = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h12_conv)
    h13_conv_b = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h12_conv)
    h13_conv = merge([h13_conv_a, h13_conv_b], mode='max', concat_axis=pr_axis)
    h13_conv = BatchNormalization(axis=pr_axis)(h13_conv)

    h14_conv_a = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h13_conv)
    h14_conv_b = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h13_conv)
    h14_conv = merge([h14_conv_a, h14_conv_b], mode='max', concat_axis=pr_axis)
    h14_conv = BatchNormalization(axis=pr_axis)(h14_conv)

    h15_conv_a = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h14_conv)
    h15_conv_b = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h14_conv)
    h15_conv = merge([h15_conv_a, h15_conv_b], mode='max', concat_axis=pr_axis)
    h15_conv = BatchNormalization(axis=pr_axis)(h15_conv)

    h16_conv_a = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h15_conv)
    h16_conv_b = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h15_conv)
    h16_conv = merge([h16_conv_a, h16_conv_b], mode='max', concat_axis=pr_axis)
    h16_conv = BatchNormalization(axis=pr_axis)(h16_conv)

    h16_pool = MaxPooling2D(pool_size=(8, 8))(h16_conv)
    h16_drop = Dropout(0.2)(h16_pool)
    # h165 = merge()
    h16 = Flatten()(h16_drop)
    h17 = MaxoutDense(500, nb_feature=5, init='glorot_uniform', W_regularizer=l2(0.0005))(h16)
    h17 = BatchNormalization(axis=1)(h17)
    h17_drop = Dropout(0.2)(h17)
    out = Dense(nb_classes, activation='softmax', init='glorot_uniform', W_regularizer=l2(0.0005))(h17_drop)

    model = Model(input=inputYUV, output=out)

    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])

    return model, batch_size
