import json

import keras.backend as K
from keras.models import model_from_json

import probe
from xsertion.data import get_cifar

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
# args = parser.parse_args()

args=lambda : None
args.p = 1.0

pstr= str(args.p).replace('.','_')
print(args.p)


nb_classes, X_train, Y_train, X_val, Y_val, X_test, Y_test = get_cifar(p=args.p, append_test=False, use_c10=True)
# X_t, Y_t, X_v, Y_v = val_split(0.2, X_train, Y_train)
print(pr_axis, input_shape)

# # nb_classes = 100
# inp = Input(shape=input_shape)
# if K.image_dim_ordering() == 'tf':
#     lb1 = Lambda(lambda x: x[:, :, :, 0:1], output_shape=(32, 32, 1))(inp)
#     lb2 = Lambda(lambda x: x[:, :, :, 1:2], output_shape=(32, 32, 1))(inp)
#     lb3 = Lambda(lambda x: x[:, :, :, 2:3], output_shape=(32, 32, 1))(inp)
# else:
#     lb1 = Lambda(lambda x: x[:, 0:1, :, :], output_shape=(1, 32, 32))(inp)
#     lb2 = Lambda(lambda x: x[:, 1:2, :, :], output_shape=(1, 32, 32))(inp)
#     lb3 = Lambda(lambda x: x[:, 2:3, :, :], output_shape=(1, 32, 32))(inp)
# input_model = Model(input=inp, output=[lb1, lb2, lb3])
#
#
# model = get_kerasnet(nb_classes, input_shape, pr_axis)
# xmodel = get_xkerasnet(nb_classes, input_shape, pr_axis)
#
nb_batch = 32
nb_epoch = 200
#
# xcnn_build=False
#
# if xcnn_build:
#     builder = XCNNBuilder(model, input_model)
#
#     # def new_con_form(nb_filter):
#     #     i = Input(shape=input_shape)
#     #     c = Convolution2D(int(math.ceil(nb_filter/2.)), 1, 1, activation='relu', name='conv_a')(i)
#     #     c = Convolution2D(int(math.ceil(nb_filter/2.)), 1, 1, activation='relu', name='conv_b')(c)
#     #     # bn = BatchNormalization(axis=pr_axis, name='bn')(c)
#     #     # a = PReLU(name='a')(bn)
#     #     # a = Dropout(.2,name='drop')(a)
#     #     m = Model(input=i, output=c)
#     #     return m
#     #
#     # builder.set_xcons(model_function=new_con_form, use_bn=True)
#     print(builder.connection_factory)
#
#     builder.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#     builder.fit(X_t, Y_t, batch_size=2*nb_batch, nb_epoch=80, validation_data=(X_v, Y_v), verbose=2)
#
#
#     single_model = builder.build_scaled_single_xcon()
#     double_model = builder.build_scaled_double_xcon()
#
#     with open('models/mtt_keras_singlea_{}.json'.format(pstr), 'w') as out:
#         out.write(single_model.to_json(indent=4))
#     with open('models/mtt_keras_doublea_{}.json'.format(pstr), 'w') as out:
#         out.write(double_model.to_json(indent=4))
#
#
# else:
with open('models/mtt_keras_singlea_{}.json'.format(pstr), 'r') as i:
    single_model = model_from_json(i.read())
# with open('models/mtt_keras_doublea_{}.json'.format(pstr), 'r') as i:
#     double_model = model_from_json(i.read())

train = True
if train:
    from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
    single_model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    reduce_LR = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, verbose=1,cooldown=5, min_lr=1e-6)
    mod_checkpoint = ModelCheckpoint('s_weights_probe_keras_best.h5', monitor='val_acc', save_best_only=True, save_weights_only=True)
    clb = [reduce_LR, mod_checkpoint]

    single_model.fit(X_train, Y_train, batch_size=nb_batch, nb_epoch=nb_epoch, validation_data=(X_test, Y_test),
                     verbose=1, callbacks=clb)
else:
    print('Load weights')
    single_model.load_weights('s_weights_probe_keras_best.h5')
    single_model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
    print(single_model.metrics_names)
    print(single_model.evaluate(X_test, Y_test))



# def md_train(model):
#     model.compile(loss='categorical_crossentropy',
#             optimizer='adam',
#             metrics=['accuracy'])
#     hist = model.fit(X_train, Y_train, batch_size=nb_batch, nb_epoch=nb_epoch, validation_data=(X_test, Y_test),
#                      verbose=2, callbacks=[lr_reducer, e_stop])
#     # accs = sorted(hist.history['val_acc'])[-10:]
#     acc = max(hist.history['val_acc'])
#     return acc

# base_acc, base_accs = md_train(model)
# # save models
# model.save_weights('baseline_weights_probe_keras.h5')
# x_acc, x_accs = md_train(xmodel)
# xmodel.save_weights('xcnn_weights_probe_keras.h5')
# single_acc, single_accs = md_train(single_model)
# single_model.save_weights('s_weights_probe_keras.h5')
# double_acc, double_accs = md_train(double_model)
# double_model.save_weights('d_weights_probe_keras.h5')

# model.load_weights('baseline_weights_probe_keras.h5')
# xmodel.load_weights('xcnn_weights_probe_keras.h5')
# single_model.load_weights('s_weights_probe_keras.h5')
# double_model.load_weights('d_weights_probe_keras.h5')


res = dict()
# ms = dict(base=model, s=single_model, d=double_model)
ms = dict(single=single_model)
for m in ms:
    mod = ms[m]
    res[m] = dict()
    for l_ind in mod.layers_by_depth:
        layer = mod.layers[l_ind]
        lname = layer.name
        if 'input' in lname or 'lambda' in lname or  'drop' in lname or 'batch' in lname:
            continue
        # get probe
        print(lname)
        p = probe.get_probe(mod, lname)
        p.summary()
        mes = probe.probe(p, X_train, Y_train, X_test, Y_test, nb_epoch=100)

        res[m][lname]=mes

with open('probe_res_keras.json', 'w') as out:
    json.dump(res, out, sort_keys=True, indent=4)
#
# builder.print_report()
# print('----------RESULTS--------')
# print("Old model:", base_acc, base_accs)
# # print("XCNN:", x_acc, x_accs)
# print('X-CNN single_model: ', single_acc, single_accs)
# print('X-CNN double_model: ', double_acc, double_accs)
single_model.summary()
for m in res:
    print(m)
    for lname in res[m]:
        print(lname, res[m][lname])

