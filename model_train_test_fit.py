import keras.backend as K
from keras.layers import Input, Lambda, Convolution2D, Dropout
from keras.models import Model, model_from_json

from xsertion import XCNNBuilder
from xsertion.data import get_cifar, val_split

input_shape = (3, 32, 32)
pr_axis = 1
if K.image_dim_ordering() == 'tf':
    input_shape = (32, 32, 3)
    pr_axis = 3

import logging
logging.basicConfig(level=logging.DEBUG)

import logging
logging.basicConfig(level=logging.DEBUG)

import argparse
parser = argparse.ArgumentParser('Tell p value')
parser.add_argument('p', help='p value', type=float)
parser.add_argument('c10', help='use c10?', type=int)
parser.add_argument('alpha', help='alpha hyper', type=float)
parser.add_argument('beta', help='beta hyper', type=float)
parser.add_argument('local', help='filter counts infered locally or globally', type=int)
args = parser.parse_args()

# args=lambda : None
# args.p = 1.0

pstr= str(args.p).replace('.','_')

use_c10 = bool(args.c10)
use_local = bool(args.local)
print('p', args.p, 'c10', use_c10, 'alpha', args.alpha, 'beta', args.beta, 'local', use_local)

nb_classes, X_train, Y_train, X_val, Y_val, X_test, Y_test = get_cifar(p=args.p, append_test=False, use_c10=use_c10)
X_t, Y_t, X_v, Y_v = val_split(0.2, X_train, Y_train)
print(input_shape,pr_axis)
from fitnet import get_fitnet, get_xfitnet
# nb_classes=100
model = get_fitnet(nb_classes, input_shape, pr_axis)
xmodel = get_xfitnet(nb_classes, input_shape, pr_axis)


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

nb_batch = 128
nb_epoch = 230

xcnn_build=True

if xcnn_build:
    builder = XCNNBuilder(model, input_model, alpha=args.alpha, beta=args.beta)
    if use_local:
        builder.set_xspot_strategy(layernames=['poolspot_1', 'poolspot_2'], filters=-1)

    def new_con_form(nb_filter):
        i = Input(shape=input_shape)
        # b = BatchNormalization(axis=pr_axis, name='bn')(i)
        d = Dropout(.25, name='d')(i)
        c = Convolution2D(nb_filter, 1, 1, activation='relu', name='c')(d)
        # c = BatchNormalization(axis=pr_axis, name='bn')(c)
        return Model(input=i, output=c)

    builder.set_xcons(model_function=new_con_form, use_bn=False)
    print(builder.connection_factory)

    builder.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    from keras.callbacks import EarlyStopping
    # reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=.1, patience=8, cooldown=3, min_lr=1e-6, verbose=2)
    e_stop = EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=15, verbose=2)

    builder.fit(X_t, Y_t, batch_size=2*nb_batch, nb_epoch=80, validation_data=(X_v, Y_v), verbose=2,
                callbacks=[e_stop])

    if args.p == 1. and not use_c10:
        builder.set_single_superlayer_accs([.5012, .3321, .3377])
    elif args.p ==.4 and not use_c10:
        builder.set_single_superlayer_accs([.3592, .2321, .2384])

    if args.p == 1. and use_c10:
        builder.set_single_superlayer_accs([.8561, .6648, .6540])
    elif args.p ==.4 and use_c10:
        builder.set_single_superlayer_accs([.7721, .5916, .5905])



    single_model = builder.build_scaled_single_xcon()

    # double_model = builder.build_scaled_double_xcon()

    with open('models/mtt_fit_singlea_{}.json'.format(pstr), 'w') as out:
        out.write(single_model.to_json(indent=4))

    # with open('models/mtt_fit_doublea_{}.json'.format(pstr), 'w') as out:
    #     out.write(double_model.to_json(indent=4))


else:
    with open('model_train_test_fit_single.json', 'r') as i:
        single_model = model_from_json(i.read())
    # with open('model_train_test_fit_double.json', 'r') as i:
    #     double_model = model_from_json(i.read())

model.name = 'baseline'
xmodel.name= 'XCNN baseline'
def md_train(model):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    lbd_ob = lambda: None
    lbd_ob.max_acc = 0.0
    import sys
    try:
        from keras.callbacks import LambdaCallback

        def a(epoch, logs):
            if logs['val_acc'] > lbd_ob.max_acc:
                lbd_ob.max_acc = logs['val_acc']

        custom_lambda_clb = LambdaCallback(on_epoch_end=a)
        hist = model.fit(X_train, Y_train, batch_size=nb_batch, nb_epoch=nb_epoch, validation_data=(X_test, Y_test),
                         verbose=2, callbacks=[custom_lambda_clb])
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
# double_acc, double_accs = md_train(double_model)

print('----------RESULTS--------')
# print("Old model:", base_acc, base_accs)
# print("XCNN:", x_acc, x_accs)
print('X-CNN single_model: ', single_acc, single_accs)
# print('X-CNN double_model: ', double_acc, double_accs)
builder.print_report()

# 1