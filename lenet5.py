from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Input


def get_model(input_shape, ch_axis, nb_classes):
    inp = Input(shape=input_shape)
    x = Convolution2D(20, 5, 5, activation='relu', border_mode='same', init='glorot_normal')(inp)
    x = MaxPooling2D(pool_size=(2,2), name='poolspot')(x)
    x = Convolution2D(50, 5, 5, activation='relu', border_mode='same', init='glorot_normal')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    x = Dense(500, activation='relu')(x)
    x = Dense(nb_classes, activation='softmax')(x)
    return Model(input=inp, output=x)










# from keras.datasets.mnist import load_data
# from keras.utils.np_utils import to_categorical
# import numpy as np
#
# (x_train, y_train), (x_test, y_test) = load_data()
# x_train = x_train[:,:,:,np.newaxis]
# x_train = x_train.astype('float64') /255.
# x_test = x_test[:,:,:,np.newaxis]
# x_test = x_test.astype('float64') /255.
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)
# print(x_train.shape, y_train.shape, x_train.dtype, y_train.dtype)

# from data import get_cifar
# nb_classes, X_train, Y_train, X_val, Y_val, X_test, Y_test = get_cifar(p=1, append_test=False, use_c10=True)
# X_train = X_train/255.
# X_test  = X_test/255.
# model = get_model(X_train.shape[1:], 3, nb_classes)
#
# model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#
# model.fit(X_train, Y_train, batch_size=128, nb_epoch=100, verbose=2, validation_data=(X_test, Y_test))


