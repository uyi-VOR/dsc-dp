import tensorflow as tf
import numpy as np
import time
import os
import uuid
import tfrecords_gen_parser
import network_model
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_model(train_tfrecords_ls,test_tfrecords_ls,type,n_epoch=3,learning_rate=1e-3,batch_size=500):

    h1 = 90
    w1 = 12

    h2 = 640
    w2 = 3

    h3 = 90
    w3 = 39*8
    c = 1

    pw = 1.0

    x1 = tf.placeholder(tf.float32, shape=[None, h1, w1, c], name='x1')
    x2 = tf.placeholder(tf.float32, shape=[None, h2, w2, c], name='x2')
    x3 = tf.placeholder(tf.float32, shape=[None, h3, w3, c], name='x3')
    # x2 = tf.placeholder(tf.float32,shape=[None, h2],name = 'x2')
    y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')
    y0 = tf.placeholder(tf.float32, [None, 2], name= 'y0')

    x11,x12,x13 = tf.split(x1,[4,4,4],axis=2)

    x21,x22,x23 = tf.split(x2,[1,1,1],axis=2)

    x31,x32,x33 = tf.split(x3,[13*8,13*8,13*8],axis=2)

    x21 = tf.reshape(x21,[-1,h2])
    x22 = tf.reshape(x22,[-1,h2])
    x23 = tf.reshape(x23,[-1,h2])


    weights = {

        'pre': tf.Variable(tf.truncated_normal([2, 30], stddev=0.01)),
        'cur': tf.Variable(tf.truncated_normal([4642, 30], stddev=0.01)),
        'next': tf.Variable(tf.truncated_normal([2, 30], stddev=0.01)),
        'out': tf.Variable(tf.truncated_normal([30, 2], stddev=0.01))
    }

    biases = {
        'b1': tf.Variable(tf.zeros([30])),
        'out': tf.Variable(tf.zeros([2]))
    }

    y2,logits = network_model.networks_model(y0,x11,x21,x31,x12,x22,x32,x13,x23,x33,weights,biases)



    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=tf.one_hot(y_,2),logits=logits,pos_weight=pw))

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    c4_, kmer_, dsc_, label_ = tfrecords_gen_parser.parse_tfrecord(train_tfrecords_ls)
    c4_batch, kmer_batch, dsc_batch, label_batch = tf.train.shuffle_batch([c4_, kmer_, dsc_, label_],batch_size=batch_size, num_threads=64, capacity=10000, min_after_dequeue=1000)

    c4_test, kmer_test, dsc_test, label_test = tfrecords_gen_parser.parse_tfrecord(test_tfrecords_ls)
    c4_batch_test, kmer_batch_test, dsc_batch_test, label_batch_test = tf.train.shuffle_batch([c4_test, kmer_test, dsc_test, label_test],batch_size=batch_size, capacity=10000, min_after_dequeue=1000,name='test_batch')


    saver = tf.train.Saver()
    start = time.clock()

    total_num = tfrecords_gen_parser.count_tfrecord_number(train_tfrecords_ls)

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
        batch_idxs = int(total_num / batch_size)
        # batch_idxs = int(2533951 / batch_size)

        for epoch in range(n_epoch):
            state0 = np.zeros((batch_size, 2))
            state1 = np.zeros((batch_size, 2))
            train_loss, train_acc, train_batch = 0, 0, 0
            val_loss, val_acc, val_batch = 0, 0, 0
            for j in range(batch_idxs):
                c4_batchs, kmer_batchs, dsc_batchs, label_batchs = sess.run([c4_batch,kmer_batch, dsc_batch, label_batch])
                _, err, ac = sess.run([train_op, loss, acc], feed_dict={x1: c4_batchs, x2: kmer_batchs, x3: dsc_batchs, y_: label_batchs, y0: state0})
                train_loss += err
                train_acc += ac
                train_batch += 1

                c4_batchs_test, kmer_batchs_test, dsc_batchs_test, label_batchs_test = sess.run([c4_batch_test, kmer_batch_test, dsc_batch_test, label_batch_test])
                err, ac = sess.run([loss, acc], feed_dict={x1: c4_batchs_test, x2: kmer_batchs_test, x3:dsc_batchs_test, y_: label_batchs_test, y0: state1})
                val_loss += err
                val_acc += ac
                val_batch += 1

                if np.mod(j, 400) == 0:
                    print("(%d/%d) train loss: %f, train acc: %f, validation loss: %f ,validation acc: %f" % (n_epoch, epoch + 1, train_loss / train_batch, train_acc / train_batch, val_loss / val_batch, val_acc / val_batch))

        end = time.clock()
        time_elaspe = end - start
        print("time elaspe: %s" % time_elaspe)

        uuid_str = uuid.uuid4().hex
        model_file = 'model/' + type + '/model_%s.ckpt' % uuid_str
        #model_file = 'model/' + type + '/model_%s.ckpt' % uuid_str

        saver.save(sess, model_file)
        coord.request_stop()
        coord.join(threads)
    tf.reset_default_graph()
    return time_elaspe, model_file, uuid_str

def test_model(model_file, test_tfrecord_ls, batch_size):
    total_num = tfrecords_gen_parser.count_tfrecord_number(test_tfrecord_ls)
    h1 = 90
    w1 = 12

    h2 = 640
    w2 = 3

    h3 = 90
    w3 = 39*8
    c = 1
    x1 = tf.placeholder(tf.float32, shape=[None, h1, w1, c], name='x1')
    x2 = tf.placeholder(tf.float32, shape=[None, h2, w2, c], name='x2')
    x3 = tf.placeholder(tf.float32, shape=[None, h3, w3, c], name='x3')
    y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')
    y0 = tf.placeholder(tf.float32, [None, 2], name='y0')

    x11, x12, x13 = tf.split(x1, [4, 4, 4], axis=2)

    x21, x22, x23 = tf.split(x2, [1, 1, 1], axis=2)

    x31, x32, x33 = tf.split(x3, [13*8, 13*8, 13*8], axis=2)
    x21 = tf.reshape(x21, [-1, h2])
    x22 = tf.reshape(x22, [-1, h2])
    x23 = tf.reshape(x23, [-1, h2])

    weights = {

        'pre': tf.Variable(tf.truncated_normal([2, 30], stddev=0.01)),
        'cur': tf.Variable(tf.truncated_normal([4642, 30], stddev=0.01)),
        'next': tf.Variable(tf.truncated_normal([2, 30], stddev=0.01)),
        'out': tf.Variable(tf.truncated_normal([30, 2], stddev=0.01))
    }

    biases = {
        'b1': tf.Variable(tf.zeros([30])),
        'out': tf.Variable(tf.zeros([2]))
    }

    y2, logits = network_model.networks_model(y0, x11, x21, x31, x12, x22, x32, x13, x23, x33, weights, biases)


    c4_test, kmer_test, dsc_test, label_test = tfrecords_gen_parser.parse_tfrecord(test_tfrecord_ls)
    c4_batch_test, kmer_batch_test, dsc_batch_test, label_batch_test = tf.train.shuffle_batch([c4_test, kmer_test, dsc_test, label_test],
                                                                              batch_size=batch_size, capacity=10000,
                                                                              min_after_dequeue=1000, name='test_batch')




    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        batch_idxs = int(total_num / batch_size)
        saver = tf.train.Saver()
        saver.restore(sess, model_file)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        ss = np.zeros((batch_size, 2))

        scores = []
        labels = []
        for i in range(batch_idxs):
            c4_batchs_test, kmer_batchs_test, dsc_batchs_test, label_batchs_test = sess.run([c4_batch_test, kmer_batch_test, dsc_batch_test, label_batch_test])
            sco = sess.run(y2, feed_dict={x1: c4_batchs_test, x2: kmer_batchs_test, x3:dsc_batchs_test, y_: label_batchs_test, y0: ss})
            scores.append(sco[:, 1])
            labels.append(label_batchs_test)
            

        scores, labels = np.asarray(scores, np.float32), np.asarray(labels, np.int32)
        scores = np.reshape(scores, (-1, 1))
        labels = np.reshape(labels, (-1, 1))
        coord.request_stop()
        coord.join(threads)
    tf.reset_default_graph()
    return scores, labels
