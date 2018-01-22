import json
import os
import sys

from xsertion.data import get_cifar

sys.setrecursionlimit(1500)

def data_getter(p, cifar10 = True):
    if not hasattr(data_getter, "cache"):
        data_getter.cache = dict()
    if p not in data_getter.cache:
        train_ratio = float(p) / 100
        num_classes, X_train, Y_train, X_val, Y_val, X_test, Y_test = get_cifar(p=train_ratio, append_test=False, use_c10=cifar10)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        data_getter.cache[p] = num_classes, (X_train, Y_train), (X_test, Y_test)
    return data_getter.cache[p]

data_getter.cache = dict()


def get_params(model):
    from keras.utils.layer_utils import count_total_params
    if hasattr(model, 'flattened_layers'):
        # Support for legacy Sequential/Merge behavior.
        flattened_layers = model.flattened_layers
    else:
        flattened_layers = model.layers
    # (trainable_params, non-trainable_params)
    return count_total_params(flattened_layers, layer_set=None)


def experiment_pass(p, model_getter, exp_name, nb_epoch, cifar10=True):
    # prep the data:
    # train_ratio = float(p)/100
    # (X_train, Y_train), (X_test, Y_test) = data_getter(p)
    if cifar10:
        hist_file = exp_name + "/{}.json".format(p)
    else:
        hist_file = exp_name + "/{}_100.json".format(p)
    if os.path.exists(hist_file) and os.path.isfile(hist_file):
        return

    train_ratio = float(p) / 100
    nb_classes, X_train, Y_train, X_val, Y_val, X_test, Y_test = get_cifar(p=train_ratio, append_test=False,
                                                                            use_c10=cifar10)
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    model, batch_size = model_getter(nb_classes)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # trainflow = ImageDataGenerator(
    #     featurewise_center=False,  # set input mean to 0 over the dataset
    #     samplewise_center=False,  # set each sample mean to 0
    #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #     samplewise_std_normalization=False,  # divide each input by its std
    #     zca_whitening=False,  # apply ZCA whitening
    #     rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    #     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    #     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    #     horizontal_flip=True,  # randomly flip images
    #     vertical_flip=False)  # randomly flip images
    #
    # # compute quantities required for featurewise normalization
    # # (std, mean, and principal components if ZCA whitening is applied)
    # trainflow.fit(X_train)
    print("Begin Training p={}".format(p))
    # experiment_history = model.fit_generator(trainflow.flow(X_train, Y_train,
    #                                  batch_size=batch_size),
    #                     samples_per_epoch=X_train.shape[0],
    #                     nb_epoch=nb_epoch,
    #                     validation_data=(X_test, Y_test), verbose=2)
    experiment_history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                                   validation_data=(X_test, Y_test), verbose=2)
                        # verbose=1,
                        # callbacks=[EarlyStopping(monitor='val_acc', min_delta=0.002, patience=20, verbose=1, mode='max')])

    s = experiment_history.history
    tp, ntp = get_params(model)
    tp = int(tp)
    # print(tp, ntp)
    s['params'] = tp
    # print(s['params'], s['val_acc'])
    with open(hist_file, 'w') as out:
        json.dump(s, out)
    macc = max(s['val_acc'])
    return tp, macc

def p_range():
    yield from [1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# def p_range():
#     yield 5

def experiment_run(exp_name, model_getter, nb_epoch):
    if not os.path.exists(exp_name) or not os.path.isdir(exp_name):
        os.mkdir(exp_name)
    results_10 = []
    results_100 = []
    for p in p_range():
        # model, batch_size = model_getter()
        tp10, macc10 = experiment_pass(p, model_getter, exp_name, nb_epoch, True)
        results_10.append((p, tp10, macc10))
        print(results_10)
        tp100, macc100 = experiment_pass(p, model_getter, exp_name, nb_epoch, False)
        results_100.append((p, tp100, macc100))
        print(results_100)
    return results_10,results_100


def main():
    # get models:
    prefix = "exp/"
    suffix = "_res"
    nb_epoch_fitnet = 230
    nb_epoch_keras  = 200
    rez = dict()
    # from exp.keras_baseline import get_baseline
    # name = prefix + "keras_baseline" + suffix
    # print(name)
    # r10, r100 = experiment_run(name, get_baseline, nb_epoch_keras)
    # rez[name]=dict(cifar10=r10, cifar100=r100)
    #
    # from exp.keras_xcnnbaseline import get_kerasxcnnbaseline
    # name = prefix + "keras_xcnn_baseline" + suffix
    # print(name)
    # r10, r100 = experiment_run(name, get_kerasxcnnbaseline, nb_epoch_keras)
    # rez[name] = dict(cifar10=r10, cifar100=r100)
    #
    # from exp.keras_xcnn_equalsplit_no_xcon import get_equalsplit_no_xcon
    # name = prefix + "keras_xcnn_eqsplit_nocon" + suffix
    # print(name)
    # r10, r100 = experiment_run(name, get_equalsplit_no_xcon, nb_epoch_keras)
    # rez[name] = dict(cifar10=r10, cifar100=r100)
    #
    # from exp.keras_xcnn_eqsplit_idcon import get_eqsplit_id_xcon
    # name = prefix + "keras_xcnn_eqsplit_idcon" + suffix
    # print(name)
    # r10, r100 = experiment_run(name, get_eqsplit_id_xcon, nb_epoch_keras)
    # rez[name] = dict(cifar10=r10, cifar100=r100)
    #
    # from exp.keras_xcnn_eqsplit_eqcon import get_eqsplit_eq_xcon
    # name = prefix + "keras_xcnn_eqsplit_eqcon" + suffix
    # print(name)
    # r10, r100 = experiment_run(name, get_eqsplit_eq_xcon, nb_epoch_keras)
    # rez[name] = dict(cifar10=r10, cifar100=r100)

    # from exp.fitnet_baseline import get_baseline
    # name = prefix + "fitnet_baseline" + suffix
    # print(name)
    # r10, r100 = experiment_run(name, get_baseline, nb_epoch_fitnet)
    # rez[name] = dict(cifar10=r10, cifar100=r100)
    #
    # from exp.fitnet_xcnn_baseline import get_fitnetxcnn
    # name = prefix + "fitnet_xcnn" + suffix
    # print(name)
    # r10, r100 = experiment_run(name, get_fitnetxcnn, nb_epoch_fitnet)
    # rez[name] = dict(cifar10=r10, cifar100=r100)

    from exp.keras_xcnn_split_nocon import get_split_no_xcon as get
    name = prefix + "keras_xcnn_split_nocon" + suffix
    print(name)
    r10, r100 = experiment_run(name, get, nb_epoch_keras)
    rez[name] = dict(cifar10=r10, cifar100=r100)

    from exp.keras_xcnn_split_eqcon import get_split_eqcon as get
    name = prefix + "keras_xcnn_split_eqcon" + suffix
    print(name)
    r10, r100 = experiment_run(name, get, nb_epoch_keras)
    rez[name] = dict(cifar10=r10, cifar100=r100)

    from exp.keras_xcnn_split_dcon import get_split_dcon as get
    name = prefix + "keras_xcnn_split_dcon" + suffix
    print(name)
    r10, r100 = experiment_run(name, get, nb_epoch_keras)
    rez[name] = dict(cifar10=r10, cifar100=r100)

    print(rez)
    with open('exp/results.json', 'w') as out:
        json.dump(rez, out)

if __name__ == "__main__":
    main()
