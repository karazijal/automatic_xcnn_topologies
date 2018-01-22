from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Input, BatchNormalization, Activation, AveragePooling2D, merge, Lambda
from keras.layers import SeparableConvolution2D
from keras.regularizers import l2

import math
def cf(x: float, eps=0.0001):
    return int(math.ceil(x - eps))

def get_resent_baseline(input_shape, nb_classes, init, reg, pr_axis, nb_stages):
    i = Input(shape=input_shape)
    c0i = Convolution2D(128, 3, 3, subsample=(2, 2), border_mode='same', bias=False, init=init, W_regularizer=l2(reg),
                        name='inp_conv')(i)
    b0i = BatchNormalization(axis=pr_axis, name='inp_bn')(c0i)
    r0i = Activation('relu', name='inp_ac')(b0i)
    xspots = []
    stage_start = r0i
    for j in range(nb_stages):
        if j > 0:
            # x = Activation('linear',name='xm_{}'.format(j))(stage_start)
            # x = MaxPooling2D(pool_size=(1, 1), name='xm_{}'.format(j))(stage_start)
            # xspots.append('xm_{}'.format(j))
            x = BatchNormalization(axis=pr_axis, name='bn_c_{}'.format(j))(stage_start)
            x = Activation('relu', name='a_c_{}'.format(j))(x)
            cs = Convolution2D(32, 1, 1, bias=False, init=init, W_regularizer=l2(reg), name='c_a_{}'.format(j))(x)
        else:
            cs = Convolution2D(32, 1, 1, bias=False, init=init, W_regularizer=l2(reg), name='c_a_{}'.format(j))(
                stage_start)
        bns = BatchNormalization(axis=pr_axis, name='bn_a_{}'.format(j))(cs)
        acs = Activation('relu', name='a_a_{}'.format(j))(bns)
        cs2 = Convolution2D(32, 3, 3, bias=False, border_mode='same', init=init, W_regularizer=l2(reg),
                            name='c_b_{}'.format(j))(acs)
        bns2 = BatchNormalization(axis=pr_axis, name='bn_b{}'.format(j))(cs2)
        acs2 = Activation('relu', name='a_b_{}'.format(j))(bns2)
        cve = Convolution2D(128, 1, 1, init=init, W_regularizer=l2(reg), name='c_c_{}'.format(j))(acs2)
        stage_start = merge([stage_start, cve], mode='sum', name='+_{}'.format(j))
    # 16 + 11*24
    bnF = BatchNormalization(axis=pr_axis, name='bn_f')(stage_start)
    rF = Activation('relu', name='a_f')(bnF)
    p = AveragePooling2D(pool_size=(16, 16), name='avp_f')(rF)
    f = Flatten()(p)
    out = Dense(nb_classes, activation='softmax', name='output_softmax')(f)
    model = Model(input=i, output=out)
    return model

def get_multilayer_resnet(input_shape, nb_classes, init, reg, pr_axis, nb_stages):
    print(init, reg, nb_stages)
    def get_slice(axis, axis_id, input_shape):
        return Lambda(
            lambda x: x[
                [slice(None) if i != axis else slice(axis_id, axis_id + 1) for i in range(len(input_shape) + 1)]],
            output_shape=[p if i + 1 != axis else 1 for i, p in enumerate(input_shape)])

    inputYUV = Input(shape=input_shape)
    input_bn = BatchNormalization(axis=pr_axis)(inputYUV)

    inputY = get_slice(pr_axis, 0, input_shape)(input_bn)
    inputU = get_slice(pr_axis, 1, input_shape)(input_bn)
    inputV = get_slice(pr_axis, 2, input_shape)(input_bn)
    s = 1.6
    mY = .5*s
    mU = .25*s
    mV = .25*s

    c0iY = Convolution2D(cf(128*mY), 3, 3, subsample=(2, 2), border_mode='same', bias=False, init=init, W_regularizer=l2(reg),
                        name='inp_convY')(inputY)
    b0iY = BatchNormalization(axis=pr_axis, name='inp_bnY')(c0iY)
    r0iY = Activation('relu', name='inp_acY')(b0iY)

    c0iU = Convolution2D(cf(128*mU), 3, 3, subsample=(2, 2), border_mode='same', bias=False, init=init, W_regularizer=l2(reg),
                         name='inp_convU')(inputU)
    b0iU = BatchNormalization(axis=pr_axis, name='inp_bnU')(c0iU)
    r0iU = Activation('relu', name='inp_acU')(b0iU)

    c0iV = Convolution2D(cf(128*mV), 3, 3, subsample=(2, 2), border_mode='same', bias=False, init=init, W_regularizer=l2(reg),
                         name='inp_convV')(inputV)
    b0iV = BatchNormalization(axis=pr_axis, name='inp_bnV')(c0iV)
    r0iV = Activation('relu', name='inp_acV')(b0iV)

    stage_startY = r0iY
    stage_startU = r0iU
    stage_startV = r0iV

    for i in range(nb_stages):
        if i > 0:
            x = BatchNormalization(axis=pr_axis, name='bnY_c_{}'.format(i))(stage_startY)
            x = Activation('relu', name='aY_c_{}'.format(i))(x)
            csY = Convolution2D(cf(32*mY), 1, 1, bias=False, init=init, W_regularizer=l2(reg), name='cY_a_{}'.format(i))(x)
            x = BatchNormalization(axis=pr_axis, name='bnU_c_{}'.format(i))(stage_startU)
            x = Activation('relu', name='aU_c_{}'.format(i))(x)
            csU = Convolution2D(cf(32 * mU), 1, 1, bias=False, init=init, W_regularizer=l2(reg),
                                name='cU_a_{}'.format(i))(x)
            x = BatchNormalization(axis=pr_axis, name='bnV_c_{}'.format(i))(stage_startV)
            x = Activation('relu', name='aV_c_{}'.format(i))(x)
            csV = Convolution2D(cf(32 * mV), 1, 1, bias=False, init=init, W_regularizer=l2(reg),
                                name='cV_a_{}'.format(i))(x)
        else:
            csY = Convolution2D(cf(32 * mY), 1, 1, bias=False, init=init, W_regularizer=l2(reg),
                                name='cY_a_{}'.format(i))(stage_startY)
            csU = Convolution2D(cf(32 * mU), 1, 1, bias=False, init=init, W_regularizer=l2(reg),
                                name='cU_a_{}'.format(i))(stage_startU)
            csV = Convolution2D(cf(32 * mV), 1, 1, bias=False, init=init, W_regularizer=l2(reg),
                                name='cV_a_{}'.format(i))(stage_startV)

        bn = BatchNormalization(axis=pr_axis, name='bnY_a_{}'.format(i))(csY)
        ac = Activation('relu', name='aY_a_{}'.format(i))(bn)
        c = Convolution2D(cf(32 * mY), 3, 3, bias=False, border_mode='same', init=init, W_regularizer=l2(reg),
                            name='cY_b_{}'.format(i))(ac)
        bn2 = BatchNormalization(axis=pr_axis, name='bnY_b{}'.format(i))(c)
        ac2 = Activation('relu', name='aY_b_{}'.format(i))(bn2)
        c2Y = Convolution2D(cf(128 * mY), 1, 1, init=init, W_regularizer=l2(reg), name='cY_c_{}'.format(i))(ac2)
        stage_startY = merge([stage_startY, c2Y], mode='sum', name='+Y_{}'.format(i))

        bn = BatchNormalization(axis=pr_axis, name='bnU_a_{}'.format(i))(csU)
        ac = Activation('relu', name='aU_a_{}'.format(i))(bn)
        c = Convolution2D(cf(32 * mU), 3, 3, bias=False, border_mode='same', init=init, W_regularizer=l2(reg),
                          name='cU_b_{}'.format(i))(ac)
        bn2 = BatchNormalization(axis=pr_axis, name='bnU_b{}'.format(i))(c)
        ac2 = Activation('relu', name='aU_b_{}'.format(i))(bn2)
        c2U = Convolution2D(cf(128 * mU), 1, 1, init=init, W_regularizer=l2(reg), name='cU_c_{}'.format(i))(ac2)
        stage_startU = merge([stage_startU, c2U], mode='sum', name='+U_{}'.format(i))

        bn = BatchNormalization(axis=pr_axis, name='bnV_a_{}'.format(i))(csV)
        ac = Activation('relu', name='aV_a_{}'.format(i))(bn)
        c = Convolution2D(cf(32 * mV), 3, 3, bias=False, border_mode='same', init=init, W_regularizer=l2(reg),
                          name='cV_b_{}'.format(i))(ac)
        bn2 = BatchNormalization(axis=pr_axis, name='bnV_b{}'.format(i))(c)
        ac2 = Activation('relu', name='aV_b_{}'.format(i))(bn2)
        c2V = Convolution2D(cf(128 * mV), 1, 1, init=init, W_regularizer=l2(reg), name='cV_c_{}'.format(i))(ac2)
        stage_startV = merge([stage_startV, c2V], mode='sum', name='+V_{}'.format(i))


    fmerge = merge([stage_startY, stage_startU, stage_startV], mode='concat', concat_axis=pr_axis)
    bnF = BatchNormalization(axis=pr_axis, name='bn_f')(fmerge)
    rF = Activation('relu', name='a_f')(bnF)
    p = AveragePooling2D(pool_size=(16, 16), name='avp_f')(rF)
    f = Flatten()(p)
    out = Dense(nb_classes, activation='softmax', name='output_softmax')(f)
    model = Model(input=inputYUV , output=out)
    return model

def get_xresnet(input_shape, nb_classes, init, reg, pr_axis, nb_stages):
    print(init, reg, nb_stages)

    def get_slice(axis, axis_id, input_shape):
        return Lambda(
            lambda x: x[
                [slice(None) if i != axis else slice(axis_id, axis_id + 1) for i in range(len(input_shape) + 1)]],
            output_shape=[p if i + 1 != axis else 1 for i, p in enumerate(input_shape)])

    inputYUV = Input(shape=input_shape)
    input_bn = BatchNormalization(axis=pr_axis)(inputYUV)

    inputY = get_slice(pr_axis, 0, input_shape)(input_bn)
    inputU = get_slice(pr_axis, 1, input_shape)(input_bn)
    inputV = get_slice(pr_axis, 2, input_shape)(input_bn)
    s = 1.6
    mY = .5 * s
    mU = .25 * s
    mV = .25 * s

    c0iY = Convolution2D(cf(128 * mY), 3, 3, subsample=(2, 2), border_mode='same', bias=False, init=init,
                         W_regularizer=l2(reg),
                         name='inp_convY')(inputY)
    b0iY = BatchNormalization(axis=pr_axis, name='inp_bnY')(c0iY)
    r0iY = Activation('relu', name='inp_acY')(b0iY)

    c0iU = Convolution2D(cf(128 * mU), 3, 3, subsample=(2, 2), border_mode='same', bias=False, init=init,
                         W_regularizer=l2(reg),
                         name='inp_convU')(inputU)
    b0iU = BatchNormalization(axis=pr_axis, name='inp_bnU')(c0iU)
    r0iU = Activation('relu', name='inp_acU')(b0iU)

    c0iV = Convolution2D(cf(128 * mV), 3, 3, subsample=(2, 2), border_mode='same', bias=False, init=init,
                         W_regularizer=l2(reg),
                         name='inp_convV')(inputV)
    b0iV = BatchNormalization(axis=pr_axis, name='inp_bnV')(c0iV)
    r0iV = Activation('relu', name='inp_acV')(b0iV)

    stage_startY = r0iY
    stage_startU = r0iU
    stage_startV = r0iV
    i = 'acc'
    Y_to_U = Convolution2D(16, 1, 1, init=init, activation='relu', name='YtoU_{}'.format(i))(stage_startY)
    Y_to_V = Convolution2D(16, 1, 1, init=init, activation='relu', name='YtoV_{}'.format(i))(stage_startY)

    U_to_Y = Convolution2D(8, 1, 1, init=init, activation='relu', name='UtoY_{}'.format(i))(stage_startU)
    U_to_V = Convolution2D(8, 1, 1, init=init, activation='relu', name='UtoV_{}'.format(i))(stage_startU)

    V_to_Y = Convolution2D(8, 1, 1, init=init, activation='relu', name='VtoY_{}'.format(i))(stage_startV)
    V_to_U = Convolution2D(8, 1, 1, init=init, activation='relu', name='VtoU_{}'.format(i))(stage_startV)

    stage_startY = merge([stage_startY, U_to_Y, V_to_Y], mode='concat', concat_axis=pr_axis, name='XY_{}'.format(i))
    stage_startU = merge([stage_startU, Y_to_U, V_to_U], mode='concat', concat_axis=pr_axis, name='XU_{}'.format(i))
    stage_startV = merge([stage_startV, Y_to_V, U_to_V], mode='concat', concat_axis=pr_axis, name='XV_{}'.format(i))

    for i in range(nb_stages):
        if i > 0:
            x = BatchNormalization(axis=pr_axis, name='bnY_c_{}'.format(i))(stage_startY)
            x = Activation('relu', name='aY_c_{}'.format(i))(x)
            csY = Convolution2D(cf(32 * mY), 1, 1, bias=False, init=init, W_regularizer=l2(reg),
                                name='cY_a_{}'.format(i))(x)
            x = BatchNormalization(axis=pr_axis, name='bnU_c_{}'.format(i))(stage_startU)
            x = Activation('relu', name='aU_c_{}'.format(i))(x)
            csU = Convolution2D(cf(32 * mU), 1, 1, bias=False, init=init, W_regularizer=l2(reg),
                                name='cU_a_{}'.format(i))(x)
            x = BatchNormalization(axis=pr_axis, name='bnV_c_{}'.format(i))(stage_startV)
            x = Activation('relu', name='aV_c_{}'.format(i))(x)
            csV = Convolution2D(cf(32 * mV), 1, 1, bias=False, init=init, W_regularizer=l2(reg),
                                name='cV_a_{}'.format(i))(x)
        else:
            csY = Convolution2D(cf(32 * mY), 1, 1, bias=False, init=init, W_regularizer=l2(reg),
                                name='cY_a_{}'.format(i))(stage_startY)
            csU = Convolution2D(cf(32 * mU), 1, 1, bias=False, init=init, W_regularizer=l2(reg),
                                name='cU_a_{}'.format(i))(stage_startU)
            csV = Convolution2D(cf(32 * mV), 1, 1, bias=False, init=init, W_regularizer=l2(reg),
                                name='cV_a_{}'.format(i))(stage_startV)

        bn = BatchNormalization(axis=pr_axis, name='bnY_a_{}'.format(i))(csY)
        ac = Activation('relu', name='aY_a_{}'.format(i))(bn)
        c = Convolution2D(cf(32 * mY), 3, 3, bias=False, border_mode='same', init=init, W_regularizer=l2(reg),
                          name='cY_b_{}'.format(i))(ac)
        bn2 = BatchNormalization(axis=pr_axis, name='bnY_b{}'.format(i))(c)
        ac2 = Activation('relu', name='aY_b_{}'.format(i))(bn2)
        c2Y = Convolution2D(cf(128 * mY), 1, 1, init=init, W_regularizer=l2(reg), name='cY_c_{}'.format(i))(ac2)
        # stage_startY = merge([stage_startY, c2Y], mode='sum', name='+Y_{}'.format(i))

        bn = BatchNormalization(axis=pr_axis, name='bnU_a_{}'.format(i))(csU)
        ac = Activation('relu', name='aU_a_{}'.format(i))(bn)
        c = Convolution2D(cf(32 * mU), 3, 3, bias=False, border_mode='same', init=init, W_regularizer=l2(reg),
                          name='cU_b_{}'.format(i))(ac)
        bn2 = BatchNormalization(axis=pr_axis, name='bnU_b{}'.format(i))(c)
        ac2 = Activation('relu', name='aU_b_{}'.format(i))(bn2)
        c2U = Convolution2D(cf(128 * mU), 1, 1, init=init, W_regularizer=l2(reg), name='cU_c_{}'.format(i))(ac2)
        # stage_startU = merge([stage_startU, c2U], mode='sum', name='+U_{}'.format(i))

        bn = BatchNormalization(axis=pr_axis, name='bnV_a_{}'.format(i))(csV)
        ac = Activation('relu', name='aV_a_{}'.format(i))(bn)
        c = Convolution2D(cf(32 * mV), 3, 3, bias=False, border_mode='same', init=init, W_regularizer=l2(reg),
                          name='cV_b_{}'.format(i))(ac)
        bn2 = BatchNormalization(axis=pr_axis, name='bnV_b{}'.format(i))(c)
        ac2 = Activation('relu', name='aV_b_{}'.format(i))(bn2)
        c2V = Convolution2D(cf(128 * mV), 1, 1, init=init, W_regularizer=l2(reg), name='cV_c_{}'.format(i))(ac2)
        # stage_startV = merge([stage_startV, c2V], mode='sum', name='+V_{}'.format(i))

        #---____Cross-connections___---#


        Y_to_U = Convolution2D(16, 1, 1, init=init, activation='relu', name='YtoU_{}'.format(i))(c2Y)
        Y_to_V = Convolution2D(16, 1, 1, init=init, activation='relu', name='YtoV_{}'.format(i))(c2Y)

        U_to_Y = Convolution2D(8, 1, 1, init=init, activation='relu', name='UtoY_{}'.format(i))(c2U)
        U_to_V = Convolution2D(8, 1, 1, init=init, activation='relu', name='UtoV_{}'.format(i))(c2U)

        V_to_Y = Convolution2D(8, 1, 1, init=init, activation='relu', name='VtoY_{}'.format(i))(c2V)
        V_to_U = Convolution2D(8, 1, 1, init=init, activation='relu', name='VtoU_{}'.format(i))(c2V)

        xmY = merge([c2Y, U_to_Y, V_to_Y], mode='concat', concat_axis=pr_axis, name='XY_{}'.format(i))
        xmU = merge([c2U, Y_to_U, V_to_U], mode='concat', concat_axis=pr_axis, name='XU_{}'.format(i))
        xmV = merge([c2V, Y_to_V, U_to_V], mode='concat', concat_axis=pr_axis, name='XV_{}'.format(i))

        stage_startY = merge([stage_startY, xmY], mode='sum', name='+Y_{}'.format(i))
        stage_startU = merge([stage_startU, xmU], mode='sum', name='+U_{}'.format(i))
        stage_startV = merge([stage_startV, xmV], mode='sum', name='+V_{}'.format(i))


    fmerge = merge([stage_startY, stage_startU, stage_startV], mode='concat', concat_axis=pr_axis)
    bnF = BatchNormalization(axis=pr_axis, name='bn_f')(fmerge)
    rF = Activation('relu', name='a_f')(bnF)
    p = AveragePooling2D(pool_size=(16, 16), name='avp_f')(rF)
    f = Flatten()(p)
    out = Dense(nb_classes, activation='softmax', name='output_softmax')(f)
    model = Model(input=inputYUV, output=out)
    return model


def get_resnet_xception(input_shape, nb_classes, init, reg, pr_axis, nb_stages):
    i = Input(shape=input_shape)
    c0i = Convolution2D(128, 3, 3, subsample=(2, 2), border_mode='same', bias=False, init=init, W_regularizer=l2(reg),
                        name='inp_conv')(i)
    b0i = BatchNormalization(axis=pr_axis, name='inp_bn')(c0i)
    r0i = Activation('relu', name='inp_ac')(b0i)
    stage_start = r0i
    for j in range(nb_stages):
        if j > 0:
            x = BatchNormalization(axis=pr_axis, name='bn_c_{}'.format(j))(stage_start)
            x = Activation('relu', name='a_c_{}'.format(j))(x)
        else:
            x = stage_start
        # cs = Convolution2D(32, 1, 1, bias=False, init=init, W_regularizer=l2(reg), name='c_a_{}'.format(j))(x)
        # bns = BatchNormalization(axis=pr_axis, name='bn_a_{}'.format(j))(cs)
        # acs = Activation('relu', name='a_a_{}'.format(j))(bns)
        # cs2 = Convolution2D(32, 3, 3, bias=False, border_mode='same', init=init, W_regularizer=l2(reg),
        #                     name='c_b_{}'.format(j))(acs)
        # bns2 = BatchNormalization(axis=pr_axis, name='bn_b{}'.format(j))(cs2)
        # acs2 = Activation('relu', name='a_b_{}'.format(j))(bns2)
        # cve = Convolution2D(128, 1, 1, init=init, W_regularizer=l2(reg), name='c_c_{}'.format(j))(acs2)
        cve  = SeparableConvolution2D(128, 3, 3, init=init, pointwise_regularizer=l2(reg), depthwise_regularizer=l2(reg),
                                      border_mode='same')(x)
        stage_start = merge([stage_start, cve], mode='sum', name='+_{}'.format(j))
    # 16 + 11*24
    bnF = BatchNormalization(axis=pr_axis, name='bn_f')(stage_start)
    rF = Activation('relu', name='a_f')(bnF)
    p = AveragePooling2D(pool_size=(16, 16), name='avp_f')(rF)
    f = Flatten()(p)
    out = Dense(nb_classes, activation='softmax', name='output_softmax')(f)
    model = Model(input=i, output=out)
    return model


def get_resnet_baseelu(input_shape, nb_classes, init, reg, pr_axis, nb_stages):
    i = Input(shape=input_shape)
    # ibn = BatchNormalization(axis=pr_axis, name='inp_bn')(i)
    c0i = Convolution2D(96, 5, 5, border_mode='same', init=init, W_regularizer=l2(reg),
                        name='inp_conv', activation='elu')(i)
    c0i2 = Convolution2D(32, 3, 3, subsample=(2, 2), border_mode='same', init=init, W_regularizer=l2(reg),
                        name='inp_conv2', activation='elu')(c0i)
    # b0i = BatchNormalization(axis=pr_axis, name='inp_bn')(c0i)
    # r0i = Activation('relu', name='inp_ac')(b0i)
    xspots = []
    stage_start = c0i2
    for j in range(nb_stages):
        if j > 0:
            # x = Activation('linear',name='xm_{}'.format(j))(stage_start)
            # x = MaxPooling2D(pool_size=(1, 1), name='xm_{}'.format(j))(stage_start)
            # xspots.append('xm_{}'.format(j))
            # x = BatchNormalization(axis=pr_axis, name='bn_c_{}'.format(j))(stage_start)
            x = Activation('elu', name='a_c_{}'.format(j))(stage_start)
            cs = Convolution2D(32, 1, 1, bias=True, init=init, W_regularizer=l2(reg), name='c_a_{}'.format(j))(x)
        else:
            cs = Convolution2D(32, 1, 1, bias=True, init=init, W_regularizer=l2(reg), name='c_a_{}'.format(j))(
                stage_start)
        # bns = BatchNormalization(axis=pr_axis, name='bn_a_{}'.format(j))(cs)
        acs = Activation('elu', name='a_a_{}'.format(j))(cs)
        cs2 = Convolution2D(32, 3, 3, bias=True, border_mode='same', init=init, W_regularizer=l2(reg),
                            name='c_b_{}'.format(j))(acs)
        # bns2 = BatchNormalization(axis=pr_axis, name='bn_b{}'.format(j))(cs2)
        acs2 = Activation('relu', name='a_b_{}'.format(j))(cs2)
        cve = Convolution2D(32, 1, 1, init=init, W_regularizer=l2(reg), name='c_c_{}'.format(j))(acs2)
        stage_start = merge([stage_start, cve], mode='sum', name='+_{}'.format(j))
    # 16 + 11*24
    # bnF = BatchNormalization(axis=pr_axis, name='bn_f')(stage_start)
    rF = Activation('elu', name='a_f')(stage_start)
    p = AveragePooling2D(pool_size=(16, 16), name='avp_f')(rF)
    f = Flatten()(p)
    out = Dense(nb_classes, activation='softmax', name='output_softmax', init=init)(f)
    model = Model(input=i, output=out)
    return model


def get_ror_resnet(input_shape, nb_classes, init, reg, pr_axis, nb_stages):
    i = Input(shape=input_shape)
    c0i = Convolution2D(128, 3, 3, subsample=(2, 2), border_mode='same', bias=False, init=init, W_regularizer=l2(reg),
                        name='inp_conv')(i)
    b0i = BatchNormalization(axis=pr_axis, name='inp_bn')(c0i)
    r0i = Activation('relu', name='inp_ac')(b0i)
    xspots = ['inp_ac']
    stage_start = r0i
    every_5 = r0i
    for j in range(nb_stages):
        if j > 0:
            # x = Activation('linear',name='xm_{}'.format(j))(stage_start)
            # x = MaxPooling2D(pool_size=(1, 1), name='xm_{}'.format(j))(stage_start)
            # xspots.append('xm_{}'.format(j))
            x = BatchNormalization(axis=pr_axis, name='bn_c_{}'.format(j))(stage_start)
            x = Activation('relu', name='a_c_{}'.format(j))(x)
            cs = Convolution2D(32, 1, 1, bias=False, init=init, W_regularizer=l2(reg), name='c_a_{}'.format(j))(x)
        else:
            cs = Convolution2D(32, 1, 1, bias=False, init=init, W_regularizer=l2(reg), name='c_a_{}'.format(j))(
                stage_start)
        bns = BatchNormalization(axis=pr_axis, name='bn_a_{}'.format(j))(cs)
        acs = Activation('relu', name='a_a_{}'.format(j))(bns)
        cs2 = Convolution2D(32, 3, 3, bias=False, border_mode='same', init=init, W_regularizer=l2(reg),
                            name='c_b_{}'.format(j))(acs)
        bns2 = BatchNormalization(axis=pr_axis, name='bn_b{}'.format(j))(cs2)
        acs2 = Activation('relu', name='a_b_{}'.format(j))(bns2)
        cve = Convolution2D(128, 1, 1, init=init, W_regularizer=l2(reg), name='c_c_{}'.format(j))(acs2)
        xspots.append('c_c_{}'.format(j))
        if (j+1) % 3==0:
            stage_start = merge([stage_start, cve, every_5], mode='sum', name='+_{}'.format(j))
            every_5 = stage_start
        else:
            stage_start = merge([stage_start, cve], mode='sum', name='+_{}'.format(j))

    # 16 + 11*24
    bnF = BatchNormalization(axis=pr_axis, name='bn_f')(stage_start)
    rF = Activation('relu', name='a_f')(bnF)
    p = AveragePooling2D(pool_size=(16, 16), name='avp_f')(rF)
    f = Flatten()(p)
    out = Dense(nb_classes, activation='softmax', name='output_softmax')(f)
    model = Model(input=i, output=out)
    return model, xspots