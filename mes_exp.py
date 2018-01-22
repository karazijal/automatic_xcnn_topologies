import argparse
import gc
import json
import os
import sys

import tensorflow as tf
from keras.layers import Input, Lambda, Dropout, Convolution2D
from keras.models import Model

import lenet5
from xsertion import XCNNBuilder
from xsertion.data import get_cifar, val_split
from xsertion.keras_interface_util import tf_ses_ctl_reset
from fitnet import get_fitnet, get_xfitnet
from kerasnet import get_xkerasnet, get_kerasnet

sys.setrecursionlimit(1500)

import keras.backend as K
input_shape = (3, 32, 32)
ch_axis = 1
if K.image_dim_ordering() == 'tf':
    input_shape = (32, 32, 3)
    ch_axis = 3

def get_modality(mode, c10, p): # use modality cashes gathered from initial experiments to speed up the proccess
    try:
        with open('{}_modalities.json'.format(mode), 'r') as modalities:
            mods = json.load(modalities)
    except:
        return None

    modl = mods[c10][p]
    if modl:
        n = len(modl)
        if n <= 2:
            return None
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

def set_modality(modality, mode, c10, p):
    try:
        with open('{}_modalities.json'.format(mode), 'r') as modalities:
            mods = json.load(modalities)
    except:
        return None

    mods[c10][p].append(modality)
    try:
        with open('{}_modalities.json'.format(mode), 'w') as modalities:
            json.dump(mods, modalities, indent=4, sort_keys=True)
    except:
        return None
    return None

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

def get_data(p, cifar_10):
    nb_classes, X_train, Y_train, X_val, Y_val, X_test, Y_test = get_cifar(p=p, append_test=False, use_c10=cifar_10)
    X_t, Y_t, X_v, Y_v = val_split(0.2, X_train, Y_train)
    return nb_classes, X_train, Y_train, X_test, Y_test, X_t, Y_t, X_v, Y_v

def get_models(mode, nb_classes, input_shape):
    if mode=='keras':
        return 32, 200, 2.0, 4.0, ['poolspot'], get_kerasnet(nb_classes, input_shape, ch_axis), get_xkerasnet(nb_classes, input_shape, ch_axis)
    elif mode=='fitnet':
        return 128, 230, 1.0, 2.0, ['poolspot_1', 'poolspot_2'], get_fitnet(nb_classes, input_shape, ch_axis), get_xfitnet(nb_classes, input_shape, ch_axis)
    elif mode=='lenet':
        return 128, 100, 2.0, 4.0, ['poolspot'], lenet5.get_model(input_shape, ch_axis, nb_classes), None

def model_train(model, X_train, Y_train, nb_batch, nb_epoch, X_test, Y_test, mode, p, use_cifar, gid, m_name):
    if model is None:
        return -1.0
    # def on_epoch_end(epoch, logs):
    #     with open('mes/hists/{m}/{t}/{p}_{c}_{g}.json'.format(m=mode, t=m_name, p=str(int(p * 100)),
    #                                                           c=(10 if use_cifar else 100), g=gid), 'w') as out:
    #         out.write(json.dumps(h, indent=4))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    hist = model.fit(X_train, Y_train, batch_size=nb_batch, nb_epoch=nb_epoch, validation_data=(X_test, Y_test),
                     verbose=2)
    acc = max(hist.history['val_acc'])
    h = hist.history
    # save the history:
    with open('mes/hists/{m}/{t}/{p}_{c}_{g}.json'.format(m=mode, t=m_name, p=str(int(p*100)), c=(10 if use_cifar else 100), g=gid), 'w') as out:
        out.write(json.dumps(h, indent=4))
    return acc

def m_save(m, mode, p, t, use_cifar, gid):
    with open('mes/models/{m}/{p}_{c}_{t}_{g}.json'.format(m=mode, p=str(int(p*100)), c=(10 if use_cifar else 100), t=t, g=gid), 'w') as out:
        out.write(m.to_json(indent=4))

def exp_run(p, mode, use_cifar10, gid, baseline=False, xcnn=False):
    directories=[ # make sure directories exists
        'mes',
        'mes/hists',
        'mes/hists/{}'.format(mode),
        'mes/hists/{}/baseline'.format(mode),
        'mes/hists/{}/xcnn'.format(mode),
        'mes/hists/{}/s'.format(mode),
        'mes/hists/{}/d'.format(mode),
        'mes/models',
        'mes/models/{}'.format(mode),
        'mes/build_hists',
        'mes/build_hists/{}'.format(mode)

    ]
    for path in directories:
        if not os.path.exists(path):
            os.mkdir(path)

    print("check", mode, p, use_cifar10, gid)

    if os.path.exists('mes/results_{m}_{g}.csv'.format(m=mode,g=gid)):
        with open('mes/results_{m}_{g}.csv'.format(m=mode,g=gid), 'r') as res_file:
            res_lines = res_file.read().strip().split('\n')
            for res_line in reversed(res_lines):
                c = int(res_line.strip().split(',')[0]) == 10
                p_val = float(res_line.strip().split(',')[1]) / 100
                if c==use_cifar10 and p_val==p:
                    return # this measurement happened

    nb_classes, X_train, Y_train, X_test, Y_test, X_t, Y_t, X_v, Y_v = get_data(p, use_cifar10)
    nb_batch, nb_epoch, alpha, beta, insert_l, model, xmodel = get_models(mode, nb_classes, input_shape)
    input_model = get_input_model(input_shape)


    print("run", mode, p, use_cifar10, gid)
    builder = XCNNBuilder(model, input_model, alpha=alpha, beta=beta)


    def new_con_form(nb_filter):
        i = Input(shape=input_shape)
        d = Dropout(.25, name='d')(i)
        c = Convolution2D(nb_filter, 1, 1, activation='relu', name='c')(d)
        return Model(input=i, output=c)

    builder.set_xcons(model_function=new_con_form, use_bn=False)
    builder.set_xspot_strategy(layernames=insert_l, filters=-1)

    sup_acc = get_modality(mode, 'c10' if use_cifar10 else 'c100', str(int(100*p)))
    if sup_acc is not None:
        builder.set_single_superlayer_accs(sup_acc)

    if mode=='lenet':
        X_train = X_train.astype('float64') /255.
        X_test = X_test.astype('float64') /255.
        X_t = X_t.astype('float64') /255.
        X_v = X_v.astype('float64') /255.

    builder.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    builder.fit(X_t, Y_t, batch_size=2 * nb_batch, nb_epoch=80, validation_data=(X_v, Y_v), verbose=2)

    # if not os.path.exists('mes/build_hists/{m}/{p}_{c}_{g}.txt'.format(m=mode, p=str(int(100*p)), c=(10 if use_cifar10 else 100), g=gid)):
    if os.path.exists('mes/build_hists/{m}/{p}_{c}_{g}.tmp'.format(m=mode, p=str(int(100*p)), c=(10 if use_cifar10 else 100), g=gid)):
        builder.load_state('mes/build_hists/{m}/{p}_{c}_{g}.tmp'.format(m=mode, p=str(int(100*p)), c=(10 if use_cifar10 else 100), g=gid))
    else:
        builder.add_persistence('mes/build_hists/{m}/{p}_{c}_{g}.tmp'.format(m=mode, p=str(int(100*p)), c=(10 if use_cifar10 else 100), g=gid))

    # TODO: add checkpointing and process recovery; also model saver and custum training callback

    single_model = builder.build()
    m_save(single_model, mode, p, 's', use_cifar10, gid)
    if sup_acc is None:
        set_modality(builder.single_mes, mode, 'c10' if use_cifar10 else 'c100', str(int(100*p)))

    # double_model = builder.build_scaled_double_xcon()
    # m_save(double_model, mode, p, 'd', use_cifar10, gid)

    with open('mes/build_hists/{m}/{p}_{c}_{g}.txt'.format(m=mode, p=str(int(100*p)), c=(10 if use_cifar10 else 100), g=gid), 'w') as out:
        out.write(builder.report())

    # remove the temp file as the build file has concluded
    os.remove('mes/build_hists/{m}/{p}_{c}_{g}.tmp'.format(m=mode, p=str(int(100*p)), c=(10 if use_cifar10 else 100), g=gid))

    if baseline:
        baseline_acc = model_train(model, X_train, Y_train, nb_batch, nb_epoch, X_test, Y_test, mode, p, use_cifar10,  gid, 'baseline')
    else:
        baseline_acc = -1
    if xcnn:
        xcnn_acc = model_train(xmodel, X_train, Y_train, nb_batch, nb_epoch, X_test, Y_test, mode, p, use_cifar10,  gid, 'xcnn')
    else:
        xcnn_acc = -1
    s_acc = model_train(single_model, X_train, Y_train, nb_batch, nb_epoch, X_test, Y_test, mode, p, use_cifar10,  gid, 's')
    # d_acc = model_train(double_model, X_train, Y_train, nb_batch, nb_epoch, X_test, Y_test, mode, p, use_cifar10,  gid, 'd')
    d_acc = -1

    commit_line = '{c},{p},{a1},{a2},{a3},{a4}\n'.format(a1=round(baseline_acc, 4), a2=round(xcnn_acc, 4), a3=round(s_acc, 4),
                                              a4=round(d_acc, 4), p=str(int(p*100)), c=(10 if use_cifar10 else 100))

    with open('mes/results_{m}_{g}.csv'.format(m=mode,g=gid), 'a') as out:
        out.write(commit_line)

    print('----------RESULTS--------')
    print("Old model:", baseline_acc)
    print("XCNN:", xcnn_acc)
    print('X-CNN single_model: ', s_acc)
    print('X-CNN double_model: ', d_acc)
    builder.print_report()
    ses_reset(mode)
    resource_check()

def resource_check():
    import resource
    curr_use = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    col = gc.collect(0)
    post_use = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    print('Mem GC run: {} collected| {} -> {}'.format(col, curr_use, post_use))

# def exp_series(mode, use_cifar10, gid):
#     # for p in [1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
#     for p in [20, 40, 60, 80, 100]:
#     # for p in [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1]:
#     # for p in [60, 70, 80, 90, 100]:
#         p = float(p) / 100
#         exp_run(p, mode, use_cifar10, gid)

def ses_reset(mode=None):
    def ses_set():
        from keras.backend.tensorflow_backend import _MANUAL_VAR_INIT, _initialize_variables
        if tf.get_default_session() is not None:
            session = tf.get_default_session()
        else:
            if not os.environ.get('OMP_NUM_THREADS'):
                config = tf.ConfigProto(allow_soft_placement=True)
            else:
                nb_thread = int(os.environ.get('OMP_NUM_THREADS'))
                config = tf.ConfigProto(intra_op_parallelism_threads=nb_thread,
                                        allow_soft_placement=True)
            if mode=='keras':
                config.gpu_options.per_process_gpu_memory_fraction = 0.44 # about half - 160
            # else:
            #     config.gpu_options.per_process_gpu_memory_fraction = 0.6
            session = tf.Session(config=config)
        if not _MANUAL_VAR_INIT:
            _initialize_variables()
        return session
    if mode=='keras':
        tf_ses_ctl_reset(ses_set)
    else:
        tf_ses_ctl_reset()


def exp_series(mode, use_cifar10, gid):
    print(mode, use_cifar10)
    if mode=='keras' and use_cifar10:
        for p, b, x in [(20, True, True),
                        (40, True, True),
                        (60, True, True),
                        (80, True, True),
                        (100, True, True)]:
            p = float(p) / 100
            exp_run(p, mode, use_cifar10, gid, baseline=b, xcnn=x)
            # tf_ses_ctl_reset()
    elif mode=='keras' and not use_cifar10:
        for p, b, x in []:
            p = float(p) / 100
            exp_run(p, mode, use_cifar10, gid, baseline=b, xcnn=x)
            # tf_ses_ctl_reset()
    elif mode=='fitnet' and use_cifar10:
        for p, b, x in [(20, True, True),
                        (30, True, True),
                        (40, True, True),
                        (50, True, True),
                        (60, True, True),
                        (70, True, True),
                        (80, True, True),
                        (90, True, True),
                        (100,True, True)]:
            p = float(p) / 100
            exp_run(p, mode, use_cifar10, gid, baseline=b, xcnn=x)
            # tf_ses_ctl_reset()
    elif mode=='fitnet' and not use_cifar10:
        for p, b, x in []:
            p = float(p) / 100
            exp_run(p, mode, use_cifar10, gid, baseline=b, xcnn=x)
            # tf_ses_ctl_reset()
    else:
        for p in [1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            p = float(p) / 100
            exp_run(p, mode, use_cifar10, gid, baseline=True, xcnn=True)
            tf_ses_ctl_reset()
def main():
    parser = argparse.ArgumentParser('Tell p value')
    parser.add_argument('mode', type=str, choices=['keras', 'fitnet', 'lenet'])
    parser.add_argument('cifar10', type=int)
    parser.add_argument('gid', type=int)
    args = parser.parse_args()
    print(args.mode, bool(args.cifar10), args.gid)

    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.85 # it grows to this much
    # set_session(tf.Session(config=config))
    ses_reset(args.mode)
    exp_series(args.mode, bool(args.cifar10), args.gid)
    exp_series(args.mode, not bool(args.cifar10), args.gid)

if __name__=='__main__':
    main()
