import json

import keras.backend as K
from keras.layers import Convolution2D
from keras.layers import Dropout, Lambda, Input, Activation
from keras.models import Model, model_from_json

from xsertion import XCNNBuilder
from xsertion.data import get_cifar, val_split
from kerasnet import get_kerasnet

input_shape = (3, 32, 32)
pr_axis = 1
if K.image_dim_ordering() == 'tf':
    input_shape = (32, 32, 3)
    pr_axis = 3

import logging
logging.basicConfig(level=logging.DEBUG)

# import argparse
# parser = argparse.ArgumentParser('Tell p value')
# parser.add_argument('p', help='p value', type=float)
# parser.add_argument('c10', help='use c10?', type=int)
# parser.add_argument('alpha', help='alpha hyper', type=float)
# parser.add_argument('beta', help='beta hyper', type=float)
# parser.add_argument('local', help='filter counts infered locally or globally', type=int)
# args = parser.parse_args()


args=lambda : None
args.p = 1
args.beta = 4.
args.alpha = 2.
args.c10 = 0
args.local = 1

pstr= str(args.p).replace('.','_')
use_c10 = bool(args.c10)
use_local = bool(args.local)
print('p', args.p, 'c10', use_c10, 'alpha', args.alpha, 'beta', args.beta, 'local', use_local)


nb_classes, X_train, Y_train, X_val, Y_val, X_test, Y_test = get_cifar(p=args.p, append_test=False, use_c10=use_c10)
X_t, Y_t, X_v, Y_v = val_split(0.2, X_train, Y_train)
print(pr_axis, input_shape)

# nb_classes = 100
inp = Input(shape=input_shape)
if K.image_dim_ordering() == 'tf':
    lb1 = Lambda(lambda x: x[:, :, :, 0:1], output_shape=(32, 32, 1))(inp)
    lb2 = Lambda(lambda x: x[:, :, :, 1:2], output_shape=(32, 32, 1))(inp)
    lb3 = Lambda(lambda x: x[:, :, :, 2:3], output_shape=(32, 32, 1))(inp)
else:
    lb1 = Lambda(lambda x: x[:, 0:1, :, :], output_shape=(1, 32, 32))(inp)
    lb2 = Lambda(lambda x: x[:, 1:2, :, :], output_shape=(1, 32, 32))(inp)
    lb3 = Lambda(lambda x: x[:, 2:3, :, :], output_shape=(1, 32, 32))(inp)
input_model = Model(input=inp, output=[lb1, lb2, lb3])


model = get_kerasnet(nb_classes, input_shape, pr_axis)
# xmodel = get_xkerasnet(nb_classes, input_shape, pr_axis)

nb_batch = 32
nb_epoch = 200

xcnn_build=True

if xcnn_build:
    builder = XCNNBuilder(model, input_model, alpha=args.alpha, beta=args.beta)
    if use_local:
        builder.set_xspot_strategy(layernames=['poolspot'], filters=-1)
    # def new_con_form(nb_filter):
    #     i = Input(shape=input_shape)
    #     # c = BatchNormalization(axis=pr_axis, name='bn_a')(i)
    #     c = Convolution2D(nb_filter*2, 1, 1, activation='relu', name='conv_a')(i)
    #     # c = BatchNormalization(axis=pr_axis, name='bn_b')(c)
    #     c = Convolution2D(nb_filter, 1, 1, activation='relu', name='conv_b')(c)
    #     # bn = BatchNormalization(axis=pr_axis, name='bn')(c)
    #     # a = PReLU(name='a')(bn)
    #     # a = Dropout(.2,name='drop')(a)
    #     m = Model(input=i, output=c)
    #     return m

    def new_con_form(nb_filter):
        i = Input(shape=input_shape)
        # b = BatchNormalization(axis=pr_axis, name='bn')(i)
        d = Dropout(.25, name='d')(i)
        c = Convolution2D(nb_filter*2, 1, 1,
                          name='c1')(d)
        # c = Activation('elu')(c)
        c = Convolution2D(nb_filter, 1, 1,
                          name='c2')(c)
        c = Activation('elu')(c)
        # c = PReLU()(c)
        # c = BatchNormalization(axis=pr_axis, name='bn')(c)
        return Model(input=i, output=c)

    builder.set_xcons(model_function=new_con_form, use_bn=False)
    print(builder.connection_factory)

    builder.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # if args.p == 1. and not use_c10:
    #     builder.set_single_superlayer_accs([.4067, .2920, .2965])
    # elif args.p ==.4 and not use_c10:
    #     builder.set_single_superlayer_accs([.2945, .2408, .2450])

    with open('keras_modalities.json', 'r') as modalities:
        mods = json.load(modalities)

    c = 'c10' if use_c10 else 'c100'
    p = str(int(args.p * 100))
    print(c, p)
    modl = mods[c][p]
    if modl:
        n = len(modl)
        m = len(modl[0])
        sup_acc = []
        for i in range(m):
            a = 0.
            for j in range(n):
                a += modl[j][i]
            a /= n
            sup_acc.append(a)
        print('Setting modalities to', sup_acc)
        builder.set_single_superlayer_accs(sup_acc)


    from keras.callbacks import ReduceLROnPlateau, EarlyStopping
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=.1, patience=12, cooldown=3, min_lr=1e-6, verbose=2)
    e_stop = EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=15, verbose=2)

    builder.fit(X_t, Y_t, batch_size=2*nb_batch, nb_epoch=80, validation_data=(X_v, Y_v), verbose=2,
                callbacks=[e_stop])
    # builder.set_single_superlayer_accs([.2968, .2182, .2082])

    single_model = builder.build()
    # builder.print_report()
    # input()
    # double_model = builder.build_scaled_double_xcon()
    # single_model.summary()
    # input()
    with open('models/mtt_keras_singlea_{}.json'.format(pstr), 'w') as out:
        out.write(single_model.to_json(indent=4))
    # with open('models/mtt_keras_doublea_{}.json'.format(pstr), 'w') as out:
    #     out.write(double_model.to_json(indent=4))


else:
    with open('model_train_test_fit_single.json', 'r') as i:
        single_model = model_from_json(i.read())
    # with open('model_train_test_fit_double.json', 'r') as i:
    #     double_model = model_from_json(i.read())


def md_train(model):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    lbd_ob = lambda : None
    lbd_ob.max_acc = 0.0
    import sys
    try:
        from keras.callbacks import LambdaCallback

        def a(epoch, logs):
            if logs['val_acc'] > lbd_ob.max_acc:
                lbd_ob.max_acc = logs['val_acc']
        custom_lambda_clb = LambdaCallback(on_epoch_end=a)
        hist = model.fit(X_train, Y_train, batch_size=nb_batch, nb_epoch=nb_epoch, validation_data=(X_test, Y_test),
                         verbose=2, callbacks=[custom_lambda_clb, reduce_lr])
    except KeyboardInterrupt:
        print("Interupt caught: curr best val_acc: {}".format(lbd_ob.max_acc))
        sys.exit(0)
    accs = sorted(hist.history['val_acc'])[-5:]
    accs = list(map(lambda x: round(x, 4), accs))
    acc = max(hist.history['val_acc'])
    print(model.name, acc)
    return acc, accs

# base_acc, base_accs = md_train(model)
# x_acc, x_accs = md_train(xmodel)
single_acc, single_accs = md_train(single_model)
print('----------RESULTS--------')
# print("Old model:", base_acc, base_accs)
# print("XCNN:", x_acc, x_accs)
print('X-CNN single_model: ', single_acc, single_accs)
# print('X-CNN double_model: ', double_acc, double_accs)
builder.print_report()
