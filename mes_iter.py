import argparse
import json
import logging

from keras.callbacks import EarlyStopping
from keras.layers import Input, Lambda, Dropout, Convolution2D
from keras.models import Model

import lenet5
from xsertion import XCNNBuilder
from xsertion.data import get_cifar, val_split
from fitnet import get_fitnet
from kerasnet import get_kerasnet

logging.basicConfig(level=logging.INFO)

import keras.backend as K
input_shape = (3, 32, 32)
ch_axis = 1
if K.image_dim_ordering() == 'tf':
    input_shape = (32, 32, 3)
    ch_axis = 3

def get_input_model(input_shape):
    inp = Input(shape=input_shape)
    if K.image_dim_ordering() == 'tf':
        lb1 = Lambda(lambda x: x[:, :, :, 0:1], output_shape=(32, 32, 1), name='Y')(inp)
        lb2 = Lambda(lambda x: x[:, :, :, 1:2], output_shape=(32, 32, 1), name='U')(inp)
        lb3 = Lambda(lambda x: x[:, :, :, 2:3], output_shape=(32, 32, 1), name='V')(inp)
    else:
        lb1 = Lambda(lambda x: x[:, 0:1, :, :], output_shape=(1, 32, 32), name='Y')(inp)
        lb2 = Lambda(lambda x: x[:, 1:2, :, :], output_shape=(1, 32, 32), name='U')(inp)
        lb3 = Lambda(lambda x: x[:, 2:3, :, :], output_shape=(1, 32, 32), name='V')(inp)
    input_model = Model(input=inp, output=[lb1, lb2, lb3])
    return input_model

def get_models(mode, nb_classes, input_shape):
    if mode=='keras':
        return 32, 200, 2.0, 4.0, ['poolspot'], get_kerasnet(nb_classes, input_shape, ch_axis), None
    elif mode=='fitnet':
        return 128, 230, 1.0, 2.0, ['poolspot_1', 'poolspot_2'], get_fitnet(nb_classes, input_shape, ch_axis), None
    elif mode=='lenet':
        return 128, 100, 2.0, 4.0, ['poolspot'], lenet5.get_model(input_shape, ch_axis, nb_classes), None

def get_data(p, cifar_10):
    nb_classes, X_train, Y_train, X_val, Y_val, X_test, Y_test = get_cifar(p=p, append_test=False, use_c10=cifar_10)
    X_t, Y_t, X_v, Y_v = val_split(0.2, X_train, Y_train)
    return nb_classes, X_train, Y_train, X_test, Y_test, X_t, Y_t, X_v, Y_v

def get_modality(mode, c10, p):
    try:
        with open('{}_modalities.json'.format(mode), 'r') as modalities:
            mods = json.load(modalities)
    except:
        return None

    modl = mods[c10][p]
    if modl:
        n = len(modl)
        # if n <= 2: # always use modalities
        #     return None
        m = len(modl[0])
        sup_acc = []
        for i in range(m):
            a = 0.
            for j in range(n):
                a += modl[j][i]
            a /= n
            sup_acc.append(a)
        print('Setting modalities to', sup_acc)
        # builder.set_single_superlayer_accs(sup_acc)
        return sup_acc
    return None

def new_con_form(nb_filter):
    i = Input(shape=input_shape)
    # b = BatchNormalization(axis=pr_axis, name='bn')(i)
    d = Dropout(.25, name='d')(i)
    c = Convolution2D(nb_filter, 1, 1, activation='relu', name='c')(d)
    return Model(input=i, output=c)

def run(p, mode, use_cifar10):
    nb_classes, X_train, Y_train, X_test, Y_test, X_t, Y_t, X_v, Y_v = get_data(p, use_cifar10)
    nb_batch, nb_epoch, alpha, beta, insert_l, model, xmodel = get_models(mode, nb_classes, input_shape)
    input_model = get_input_model(input_shape)

    print("run", mode, p, use_cifar10)
    builder = XCNNBuilder(model, input_model, alpha=alpha, beta=beta)
    builder.set_xcons(model_function=new_con_form, use_bn=False)
    builder.set_xspot_strategy(layernames=insert_l, filters=-1)
    # nb_batch=1
    # nb_epoch=1
    sup_acc = get_modality(mode, 'c10' if use_cifar10 else 'c100', str(int(100 * p)))
    if sup_acc is not None:
        builder.set_single_superlayer_accs(sup_acc)

    if mode == 'lenet':
        X_train = X_train.astype('float64') / 255.
        X_test = X_test.astype('float64') / 255.
        X_t = X_t.astype('float64') / 255.
        X_v = X_v.astype('float64') / 255.

    builder.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    builder.fit(X_t, Y_t, batch_size=2 * nb_batch, nb_epoch=80, validation_data=(X_v, Y_v), verbose=2)
    builder.build() # incase there is no modalities
    builder.fit(X_t, Y_t, batch_size=2 * nb_batch, nb_epoch=80, validation_data=(X_v, Y_v), verbose=1,
                callbacks=[EarlyStopping(monitor='val_acc', min_delta=5e-4, patience=10)])

    iter_model = builder.build_iter(.05, iter_max=12, rep=2, final_weight_transfer=True)

    iter_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    lbd_ob = lambda : None
    lbd_ob.max_acc = 0.0
    lbd_ob.epoch = -1
    import sys
    try:
        from keras.callbacks import LambdaCallback
        def a(epoch, logs):
            if logs['val_acc'] > lbd_ob.max_acc:
                lbd_ob.max_acc = logs['val_acc']
                lbd_ob.epoch = epoch
        custom_lambda_clb = LambdaCallback(on_epoch_end=a)
        hist = iter_model.fit(X_train, Y_train, batch_size=nb_batch, nb_epoch=nb_epoch, validation_data=(X_test, Y_test),
                         verbose=2, callbacks=[custom_lambda_clb])
    except KeyboardInterrupt:
        print("Interupt caught: curr best val_acc: {} ({})".format(lbd_ob.max_acc, lbd_ob.epoch))
        sys.exit(0)
    max_acc = max(hist.history['val_acc'])
    max_acc_epoch = hist.history['val_acc'].index(max_acc) + 1

    iter_model.summary()
    builder.print_report()
    print('-#-#-#-:  Acc: {} at {}'.format(max_acc, max_acc_epoch))

    


def main():
    parser = argparse.ArgumentParser('Tell p value')
    parser.add_argument('p', type=float)
    parser.add_argument('mode', type=str, choices=['keras', 'fitnet', 'lenet'])
    parser.add_argument('cifar10', type=int)
    # parser.add_argument('gid', type=int)
    args = parser.parse_args()
    print(args.p, args.mode, bool(args.cifar10))
    run(args.p, args.mode, bool(args.cifar10))

if __name__=='__main__':
    main()
