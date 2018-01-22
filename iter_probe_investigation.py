from keras.layers import Input, Lambda, BatchNormalization, Convolution2D, MaxPooling2D, Dropout, merge, Flatten, Dense
from keras.models import Model

from xsertion.data import get_cifar


def get_slice(axis, axis_id, input_shape):
    return Lambda(
        lambda x: x[[slice(None) if i != axis else slice(axis_id, axis_id + 1) for i in range(len(input_shape) + 1)]],
        output_shape=[p if i+1 != axis else 1 for i, p in enumerate(input_shape)])

def get_model(input_shape, pr_axis, nb_classes):
    inputYUV = Input(shape=input_shape)
    # inputNorm = BatchNormalization(axis=1)(inputYUV)

    inputY = get_slice(pr_axis, 0, input_shape)(inputYUV)
    inputU = get_slice(pr_axis, 1, input_shape)(inputYUV)
    inputV = get_slice(pr_axis, 2, input_shape)(inputYUV)

    inputYnorm = BatchNormalization(axis=pr_axis)(inputY)
    inputUnorm = BatchNormalization(axis=pr_axis)(inputU)
    inputVnorm = BatchNormalization(axis=pr_axis)(inputV)

    convY = Convolution2D(21, 3, 3, border_mode='same', activation='relu')(inputYnorm)
    convU = Convolution2D(21, 3, 3, border_mode='same', activation='relu')(inputUnorm)
    convV = Convolution2D(21, 3, 3, border_mode='same', activation='relu')(inputVnorm)

    convY = Convolution2D(21, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(21, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(21, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    # ------------------

    # Y_to_U = Convolution2D(21, 1, 1, border_mode='same', activation='relu')(poolY)
    # Y_to_V = Convolution2D(21, 1, 1, border_mode='same', activation='relu')(poolY)
    #
    # U_to_Y = Convolution2D(21, 1, 1, border_mode='same', activation='relu')(poolU)
    # U_to_V = Convolution2D(21, 1, 1, border_mode='same', activation='relu')(poolU)
    #
    # V_to_Y = Convolution2D(21, 1, 1, border_mode='same', activation='relu')(poolV)
    # V_to_U = Convolution2D(21, 1, 1, border_mode='same', activation='relu')(poolV)
    #
    # Ymap = merge([poolY, U_to_Y, V_to_Y], mode='concat', concat_axis=pr_axis)
    # Umap = merge([poolU, V_to_U, Y_to_U], mode='concat', concat_axis=pr_axis)
    # Vmap = merge([poolV, Y_to_V, U_to_V], mode='concat', concat_axis=pr_axis)

    Ycon = poolY
    Ucon = poolU
    Vcon = poolV

    # ------------------

    convY = Convolution2D(42, 3, 3, border_mode='same', activation='relu')(Ycon)
    convU = Convolution2D(42, 3, 3, border_mode='same', activation='relu')(Ucon)
    convV = Convolution2D(42, 3, 3, border_mode='same', activation='relu')(Vcon)

    convY = Convolution2D(42, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(42, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(42, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    concatenate_map = merge([poolY, poolU, poolV], mode='concat', concat_axis=pr_axis)

    reshape = Flatten()(concatenate_map)
    fc = Dense(512, activation='relu')(reshape)
    fc = Dropout(0.5)(fc)
    out = Dense(nb_classes, activation='softmax')(fc)

    model= Model(input=inputYUV, output=out)
    return model, inputYUV

def get_model2(input_shape, pr_axis, nb_classes):
    inputYUV = Input(shape=input_shape)
    # inputNorm = BatchNormalization(axis=1)(inputYUV)

    inputY = get_slice(pr_axis, 0, input_shape)(inputYUV)
    inputU = get_slice(pr_axis, 1, input_shape)(inputYUV)
    inputV = get_slice(pr_axis, 2, input_shape)(inputYUV)

    inputYnorm = BatchNormalization(axis=pr_axis)(inputY)
    inputUnorm = BatchNormalization(axis=pr_axis)(inputU)
    inputVnorm = BatchNormalization(axis=pr_axis)(inputV)

    convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(inputYnorm)
    convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputUnorm)
    convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputVnorm)

    convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    # ------------------

    # Y_to_U = Convolution2D(21, 1, 1, border_mode='same', activation='relu')(poolY)
    # Y_to_V = Convolution2D(21, 1, 1, border_mode='same', activation='relu')(poolY)
    #
    # U_to_Y = Convolution2D(21, 1, 1, border_mode='same', activation='relu')(poolU)
    # U_to_V = Convolution2D(21, 1, 1, border_mode='same', activation='relu')(poolU)
    #
    # V_to_Y = Convolution2D(21, 1, 1, border_mode='same', activation='relu')(poolV)
    # V_to_U = Convolution2D(21, 1, 1, border_mode='same', activation='relu')(poolV)
    #
    # Ymap = merge([poolY, U_to_Y, V_to_Y], mode='concat', concat_axis=pr_axis)
    # Umap = merge([poolU, V_to_U, Y_to_U], mode='concat', concat_axis=pr_axis)
    # Vmap = merge([poolV, Y_to_V, U_to_V], mode='concat', concat_axis=pr_axis)

    Ycon = poolY
    Ucon = poolU
    Vcon = poolV

    # ------------------

    convY = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(Ycon)
    convU = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(Ucon)
    convV = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(Vcon)

    convY = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    concatenate_map = merge([poolY, poolU, poolV], mode='concat', concat_axis=pr_axis)

    reshape = Flatten()(concatenate_map)
    fc = Dense(512, activation='relu')(reshape)
    fc = Dropout(0.5)(fc)
    out = Dense(nb_classes, activation='softmax')(fc)

    model = Model(input=inputYUV, output=out)
    return model, inputYUV


def probe(model, layer_name, input_tensor, nb_clases, *args, **kwargs):
    for layer in model.layers:
        layer.trainable = False
    # model.compile(*args, **kwargs)

    attach_tensor = model.get_layer(name=layer_name).output
    print(attach_tensor)

    # define probe
    f = Flatten()(attach_tensor)
    d = Dense(nb_clases, activation='softmax')(f)
    prob = Model(input_tensor, d)
    prob.compile(*args, **kwargs)
    return prob

nb_classes, X_train, Y_train, X_val, Y_val, X_test, Y_test = get_cifar(p=1, append_test=False, use_c10=True)
# X_t, Y_t, X_v, Y_v = val_split(0.2, X_train, Y_train)

def md_train(model, nb_batch=32, nb_epoch=80):
    model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
    hist = model.fit(X_train, Y_train, batch_size=nb_batch, nb_epoch=nb_epoch, validation_data=(X_test, Y_test), verbose=2)
    accs = sorted(hist.history['val_acc'])[-10:]
    acc = max(accs)
    return acc, accs




model, inp = get_model2((32,32,3), 3, 10)
train = False
if train:
    acc, _ = md_train(model, 32, 30)
    model.summary()
    print(acc)
    model.save_weights('keras2_iter_approach_weights.h5')
else:
    model.load_weights('keras2_iter_approach_weights.h5')

# p = probe(model, 'lambda_1', input, 10, loss='categorical_crossentropy',
#                                         optimizer='adam',
#                                         metrics=['accuracy'])

# print(model.input)
# print(model.output.shape[1])
#
for layer in model.layers:
    print(layer.name)
input()
# print(md_train(p))
res = []
for l_name in ['lambda_1', 'lambda_2', 'lambda_3',
               'maxpooling2d_1', 'maxpooling2d_2', 'maxpooling2d_3',
               'maxpooling2d_4', 'maxpooling2d_5', 'maxpooling2d_6']:
    p = probe(model, l_name, inp, 10, loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])
    p.summary()
    acc, accs = md_train(p, nb_batch=32, nb_epoch=15)
    res.append((l_name, acc))

for r in res:
    print(r)