import keras.backend as K
from keras.layers import Lambda, Input
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
logging.basicConfig(level=logging.DEBUG)

# import argparse
# parser = argparse.ArgumentParser('Tell p value')
# parser.add_argument('p', help='p value', type=float)
# args = parser.parse_args()

args=lambda : None
args.p = 0.01

pstr= str(args.p).replace('.','_')
print(args.p)


nb_classes, X_train, Y_train, X_val, Y_val, X_test, Y_test = get_cifar(p=args.p, append_test=False, use_c10=True)
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
restore = True

if xcnn_build:
    builder = XCNNBuilder(model, input_model)

    # def new_con_form(nb_filter):
    #     i = Input(shape=input_shape)
    #     c = Convolution2D(int(math.ceil(nb_filter/2.)), 1, 1, activation='relu', name='conv_a')(i)
    #     c = Convolution2D(int(math.ceil(nb_filter/2.)), 1, 1, activation='relu', name='conv_b')(c)
    #     # bn = BatchNormalization(axis=pr_axis, name='bn')(c)
    #     # a = PReLU(name='a')(bn)
    #     # a = Dropout(.2,name='drop')(a)
    #     m = Model(input=i, output=c)
    #     return m
    #
    # builder.set_xcons(model_function=new_con_form, use_bn=True)
    # print(builder.connection_factory)

    builder.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    builder.fit(X_t, Y_t, batch_size=2*nb_batch, nb_epoch=1, validation_data=(X_v, Y_v), verbose=2)
    if restore:
        builder.load_state('build_per.tmp')
    else:
        builder.add_persistence('build_per.tmp')
    single_model = builder.build_scaled_single_xcon()
    double_model = builder.build_scaled_double_xcon()

    with open('models/mtt_keras_singlea_{}.json'.format(pstr), 'w') as out:
        out.write(single_model.to_json(indent=4))
    with open('models/mtt_keras_doublea_{}.json'.format(pstr), 'w') as out:
        out.write(double_model.to_json(indent=4))


else:
    with open('model_train_test_fit_single.json', 'r') as i:
        single_model = model_from_json(i.read())
    with open('model_train_test_fit_double.json', 'r') as i:
        double_model = model_from_json(i.read())


def md_train(model):
    model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
    hist = model.fit(X_train, Y_train, batch_size=nb_batch, nb_epoch=nb_epoch, validation_data=(X_test, Y_test), verbose=2)
    accs = sorted(hist.history['val_acc'])[-10:]
    acc = max(accs)
    return acc, accs

base_acc, base_accs = md_train(model)
x_acc, x_accs = md_train(xmodel)
single_acc, single_accs = md_train(single_model)
double_acc, double_accs = md_train(double_model)

print('----------RESULTS--------')
print("Old model:", base_acc, base_accs)
print("XCNN:", x_acc, x_accs)
print('X-CNN single_model: ', single_acc, single_accs)
print('X-CNN double_model: ', double_acc, double_accs)
builder.print_report()
