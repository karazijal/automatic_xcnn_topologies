from keras.models import Model
from keras.layers import Flatten, Dense, BatchNormalization, Dropout

def get_probe(model, layer_name):
    for layer in model.layers:
        layer.trainable = False

    input_tensor = model.input
    attach_tensor = model.get_layer(name=layer_name).output
    nb_classes = int(model.output.shape[1])
    # print(nb_classes)


    # define probe
    if len(attach_tensor.shape) >= 3:
        bn = BatchNormalization(axis=3, name="pbn")(attach_tensor)
        f = Flatten(name='pflat')(bn)
    else:
        f = BatchNormalization(axis=1,name="pbn")(attach_tensor)
        # f = attach_tensor
    drop = Dropout(.2, name='pdrop')(f)
    d = Dense(nb_classes, activation='softmax', name='psoft')(drop)
    prob = Model(input_tensor, d)

    return prob


from keras.callbacks import ReduceLROnPlateau, EarlyStopping

lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, verbose=1, cooldown=4, min_lr=1e-7)
# e_stop = EarlyStopping(monitor='val_acc', min_delta=0.0002, patience=15, verbose=1)

def probe(probe, X_train, Y_train, X_test, Y_test, nb_batch=32, nb_epoch=80):
    probe.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    hist = probe.fit(X_train, Y_train, batch_size=nb_batch, nb_epoch=nb_epoch, validation_data=(X_test, Y_test),
                     verbose=2, callbacks=[lr_reducer])
    # accs = sorted(hist.history['val_acc'])[-10:]
    # acc = max(accs)
    mes = max(hist.history['val_acc'])
    print(mes)
    return mes