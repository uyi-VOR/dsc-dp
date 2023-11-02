import tensorflow as tf
import numpy as np


def rnn_basic(_x,_w1,_w2,_b1,_b2):
    logits = tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(_x, _w1), _b1)), _w2) + _b2
    _y = tf.nn.softmax(logits)
    return _y,logits

def rnn_output(_s1,_x,_s2,_w,_w2,_b1,_b2):
    _sxs = tf.concat([_s1,_x,_s2],1)
    output,lgts = rnn_basic(_sxs,_w,_w2,_b1,_b2)
    return output,lgts

def rnn_state(_e1,_e2, _w, _w2,_b1,_b2):
    _e12 = tf.concat([_e1,_e2],1)
    state,_ = rnn_basic(_e12,_w,_w2,_b1,_b2)
    return state

def networks_model(s,x11,x21,x31,x12,x22,x32,x13,x23,x33, _weight,_biase):

    _x1 = cnn_merge(x11,x21,x31)
    _x2 = cnn_merge(x12,x22,x32)
    _x3 = cnn_merge(x13,x23,x33)


    w_pc = tf.concat([_weight['pre'],_weight['cur']],0)
    w_cn = tf.concat([_weight['cur'],_weight['next']],0)
    w_pcn = tf.concat([w_pc,_weight['next']],0)


    _s2 = rnn_state(s,_x1,w_pc,_weight['out'],_biase['b1'],_biase['out'])
    # _s3 = rnn_state(_s2,_x2,w_pc,_weight['out'],_biase['b1'],_biase['out'])

    s2_ = rnn_state(_x3,s,w_cn,_weight['out'],_biase['b1'],_biase['out'])
    # s1_ = rnn_state(_x2,s2_,w_cn,_weight['out'],_biase['b1'],_biase['out'])

    # _y1,_ = rnn_output(s,_x1,s1_,w_pcn,_weight['out'],_biase['b1'],_biase['out'])
    _y2, lgts = rnn_output(_s2,_x2,s2_,w_pcn,_weight['out'],_biase['b1'],_biase['out'])
    # _y3, _ = rnn_output(_s3,_x3,s,w_pcn,_weight['out'],_biase['b1'],_biase['out'])

    return _y2,lgts

def cnn_merge(_x1,_x2,_x3):

#对_x1进行两次卷积池化
    conv31 = tf.layers.conv2d(inputs=_x1, filters=250, kernel_size=[7, 2], padding="valid", activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool31 = tf.layers.max_pooling2d(inputs=conv31, pool_size=[2, 1], strides=2)
    drop31 = tf.layers.dropout(pool31, 0.3)

    conv32 = tf.layers.conv2d(inputs=drop31, filters=200, kernel_size=[3, 1], padding="valid",
                              activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool32 = tf.layers.max_pooling2d(inputs=conv32, pool_size=[2, 1], strides=2)
    drop32 = tf.layers.dropout(pool32, 0.3)

    # re31 = tf.reshape(drop32, [-1, 20*200])
    re31 = tf.reshape(drop32, [-1, 20*200])
# 对x3进行两次卷积池化
    conv33 = tf.layers.conv2d(inputs=_x3, filters=250, kernel_size=[7, 104], padding="valid", activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool33 = tf.layers.max_pooling2d(inputs=conv33, pool_size=[2, 1], strides=2)
    drop33 = tf.layers.dropout(pool33, 0.3)

    conv34 = tf.layers.conv2d(inputs=drop33, filters=200, kernel_size=[3, 1], padding="valid",
                              activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool34 = tf.layers.max_pooling2d(inputs=conv34, pool_size=[2, 1], strides=2)
    drop34 = tf.layers.dropout(pool34, 0.3)

    conv35 = tf.layers.conv2d(inputs=drop34, filters=2, kernel_size=[1, 1], padding="valid",
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool35 = tf.layers.max_pooling2d(inputs=conv35, pool_size=[20, 1], strides=1)
    drop35 = tf.layers.dropout(pool35, 0.3)


    re32 = tf.reshape(drop35,[-1,2])
# 将x1,x2,x3进行拼接
    co1 = tf.concat([re31,_x2,re32], axis=1)
    return co1