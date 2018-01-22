import json

import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Convolution2D
from keras.layers import Dropout, Lambda, Input
from keras.models import Model, model_from_json

from xsertion import XCNNBuilder
from xsertion.data import get_cifar, val_split
from kerasnet import get_kerasnet, get_xkerasnet

input_shape = (3, 32, 32)
pr_axis = 1
if K.image_dim_ordering() == 'tf':
    input_shape = (32, 32, 3)
    pr_axis = 3

import logging
logging.basicConfig(level=logging.INFO)

# import argparse
# parser = argparse.ArgumentParser('Tell p value')
# parser.add_argument('p', help='p value', type=float)
# args = parser.parse_args()

args=lambda : None
args.p = 1.0

pstr= str(args.p).replace('.','_')
print(args.p)

use_c10 = False
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
xmodel = get_xkerasnet(nb_classes, input_shape, pr_axis)

nb_batch = 32
nb_epoch = 200

xcnn_build=True

if xcnn_build:
    builder = XCNNBuilder(model, input_model, alpha=2, beta=4)
    builder.set_xspot_strategy(layernames=['poolspot'], filters=-1)

    def new_con_form(nb_filter):
        i = Input(shape=input_shape)
        # b = BatchNormalization(axis=pr_axis, name='bn')(i)
        d = Dropout(.25, name='d')(i)
        c = Convolution2D(nb_filter, 1, 1, activation='relu', name='c')(d)
        return Model(input=i, output=c)

    builder.set_xcons(model_function=new_con_form, use_bn=False)
    print(builder.connection_factory)

    builder.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    builder.fit(X_t, Y_t, batch_size=2*nb_batch, nb_epoch=80, validation_data=(X_v, Y_v), verbose=2,
                callbacks=[EarlyStopping(monitor='val_acc', min_delta=3e-4, patience=10)])

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

    single_model = builder.build()
    with open('models/mtt_keras_singlea_{}.json'.format(pstr), 'w') as out:
        out.write(single_model.to_json(indent=4))

    iter_model_t= builder.build_iter(.1, iter_max=15, rep=3, final_weight_transfer=True)
    with open('models/mtt_keras_itera_{}.json'.format(pstr), 'w') as out:
        out.write(iter_model.to_json(indent=4))
    single_model = builder.build()
else:
    with open('model_train_test_fit_single.json', 'r') as i:
        single_model = model_from_json(i.read())
    with open('model_train_test_fit_iter.json', 'r') as i:
        double_model = model_from_json(i.read())


def md_train(model):
    model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
    hist = model.fit(X_train, Y_train, batch_size=nb_batch, nb_epoch=nb_epoch, validation_data=(X_test, Y_test), verbose=2)
    accs = sorted(hist.history['val_acc'])[-10:]
    acc = max(accs)
    return acc, accs

# base_acc, base_accs = md_train(model)

# single_acc, single_accs = md_train(single_model) # no need to train this as already have these measurements
# double_acc, double_accs = md_train(iter_model)
double_acc1, double_accs1 = md_train(iter_model_t)

print('----------RESULTS--------')
# print("Old model:", base_acc, base_accs)
# print("XCNN:", x_acc, x_accs)
# print('X-CNN single_model: ', single_acc)
# print('X-CNN iter_model: ', double_acc)
print('X-CNN iter_model1:', double_acc1)
builder.print_report()
