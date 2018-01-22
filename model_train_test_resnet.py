import keras.backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Convolution2D
from keras.layers import Dropout, Lambda, Input
from keras.models import Model, model_from_json
from keras.regularizers import l2

from xsertion import XCNNBuilder
from xsertion.data import get_cifar, val_split

input_shape = (3, 32, 32)
pr_axis = 1
if K.image_dim_ordering() == 'tf':
    input_shape = (32, 32, 3)
    pr_axis = 3
import logging
logging.basicConfig(level=logging.DEBUG)

def get_params(model):
    from keras.utils.layer_utils import count_total_params
    if hasattr(model, 'flattened_layers'):
        # Support for legacy Sequential/Merge behavior.
        flattened_layers = model.flattened_layers
    else:
        flattened_layers = model.layers
    # (trainable_params, non-trainable_params)
    return count_total_params(flattened_layers, layer_set=None)

# import argparse
# parser = argparse.ArgumentParser('Tell p value')
# parser.add_argument('p', help='p value', type=float)
# args = parser.parse_args()
def args():
    pass
args.p =1.0
pstr= str(args.p).replace('.','_')
print(args.p)

nb_classes, X_train, Y_train, X_val, Y_val, X_test, Y_test = get_cifar(p=args.p, append_test=False, use_c10=False)
X_t, Y_t, X_v, Y_v = val_split(0.2, X_train, Y_train)
print(pr_axis, input_shape)
#
mean_image = X_train.mean(axis=0)
std_image = X_train.std(axis=0)
X_TRAIN = (X_train - mean_image)/std_image
X_TEST = (X_test - mean_image)/std_image
# print(X_TRAIN.shape)
# print(X_TEST.shape)

init ='he_normal'
reg = 0.0001

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

# from resnet import get_resent_baseline
# model = get_resent_baseline(input_shape, nb_classes, init, reg, pr_axis, 10)
from resnet import get_ror_resnet
model, xspots = get_ror_resnet(input_shape, nb_classes, init, reg, pr_axis, 12)
model.summary()

nb_batch = 64
nb_epoch = 200

from keras.callbacks import LearningRateScheduler

def lr_rate_drop(epoch):
    if epoch > 75:
        return 1.e-5
    elif epoch > 50:
        return 1.e-4
    else:
        return 1.e-3

clb = [LearningRateScheduler(lr_rate_drop)]



xcnn_build=True
#
if xcnn_build:
    builder = XCNNBuilder(model, input_model, alpha=2.0, beta=2.0)
    builder.set_xspot_strategy(layernames=xspots, filters=-1)

    # def new_con_form(nb_filter, **kwargs):
    #     assert kwargs and 'inbound_name' in kwargs
    #     i = Input(shape=input_shape)
    #     x = i
    #     # if 'inp_ac' not in kwargs['inbound_name']:
    #     x = BatchNormalization(axis=pr_axis, name='bn_pre')(x)
    #     x = Activation('relu', name='a_pre')(x)
    #     x = Convolution2D(int(math.ceil(nb_filter / 2.)), 1, 1, activation='relu', name='conv_a', init=init,
    #                       W_regularizer=l2(reg))(x)
    #     x = Convolution2D(int(math.ceil(nb_filter / 2.)), 1, 1, activation='relu', name='conv_b', init=init,
    #                       W_regularizer=l2(reg))(x)
    #     # bn = BatchNormalization(axis=pr_axis, name='bn')(c)
    #     # a = Activation('relu', name='a')(bn)
    #     # a = Dropout(.2, name='drop')(a)
    #     m = Model(input=i, output=x)
    #     return m
    def new_con_form(nb_filter):
        i = Input(shape=input_shape)
        d = Dropout(.2, name='d')(i)
        c = Convolution2D(nb_filter*2, 1, 1, activation='elu', name='c1', init=init, W_regularizer=l2(reg))(d)
        c = Convolution2D(nb_filter, 1, 1, activation='elu', name='c2', init=init, W_regularizer=l2(reg))(c)
        return Model(input=i, output=c)

    builder.set_xcons(model_function=new_con_form, use_bn=False)
    # builder.set_xspot_strategy(layernames=xspots, filters=128)
    # builder.set_xcons(model_function=new_con_form, use_bn=False)
    # builder.set_xcons(use_bn=False, activation='relu', init=init, W_regularizer=l2(reg), bias=False)
    # builder.set_xcons(model_function=new_con_form, use_bn=False)
    builder.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    builder.fit(X_t, Y_t, batch_size=nb_batch, nb_epoch=80, validation_data=(X_v, Y_v), verbose=2,
                callbacks=clb)
    # builder.set_single_superlayer_accs([0.8054, 0.6278, 0.6116])
    # builder.set_single_superlayer_accs([0.735, 0.5686, 0.5711])
    # [0.735, 0.5686, 0.5711]
    # builder.set_lane_scale(1.6875)
    # builder.set_single_superlayer_accs([.84, .77, .79])
    single_model = builder.build()
    # single_model_nocon = builder.build_single_no_con(priors=priors)
    # double_model = builder.build_scaled_double_xcon()
    # single_model_nocon = builder.build_single_no_con()

    with open('models/mtt_resnet_singlea_{}.json'.format(pstr), 'w') as out:
        out.write(single_model.to_json(indent=4))
    # with open('models/mtt_resnet_doublea_{}.json'.format(pstr), 'w') as out:
    #     out.write(double_model.to_json(indent=4))


else:
    with open('models/mtt_resnet_singlea_{}.json', 'r') as i:
        single_model = model_from_json(i.read())
    # with open('models/mtt_resnet_doublea_{}.json', 'r') as i:
    #     double_model = model_from_json(i.read())
def md_train(model):
    model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
    hist = model.fit(X_train, Y_train, batch_size=nb_batch, nb_epoch=nb_epoch, validation_data=(X_test, Y_test),
                     callbacks=[ReduceLROnPlateau('val_acc', factor=0.1, patience=15, cooldown=5, min_lr=1e-6, verbose=2)],
                     verbose=2)
    accs = sorted(hist.history['val_acc'])[-10:]
    acc = max(accs)
    return acc, accs

def md_train1(model):
    model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
    hist = model.fit(X_train, Y_train, batch_size=nb_batch, nb_epoch=100, validation_data=(X_test, Y_test),
                     callbacks=clb,
                     verbose=2)
    accs = sorted(hist.history['val_acc'])[-10:]
    acc = max(accs)
    return acc, accs

def md_train_data_gen(model):
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(X_TRAIN)
    model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
    hist = model.fit_generator(datagen.flow(X_TRAIN, Y_train,
                        batch_size=nb_batch),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch*3,
                        validation_data=(X_TEST, Y_test),
                        verbose=2,
                        callbacks=[ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=10, verbose=1, cooldown=5, min_lr=0.000001)])
    accs = sorted(hist.history['val_acc'])[-10:]
    acc = max(accs)
    return acc, accs

# mod_acc, _ = md_train1(model)
sing_acc, _ = md_train(single_model)
# doub_acc, _ = md_train(double_model)

print('----------RESULTS--------')
# print("Old model:", mod_acc)
print('X-CNN single_model: ', sing_acc)
# print('X-CNN double_model: ', doub_acc)
# builder.print_report()
