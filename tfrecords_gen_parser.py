from sklearn.calibration import label_binarize
import tensorflow as tf
from discrete_model import kmer_featurization
import sequential_model
import dnashape_model

COUNT = 0

def count_tfrecord_number(tf_records_ls):
    c = 0
    for fn in tf_records_ls:
        for record in tf.python_io.tf_record_iterator(fn):
            c += 1
    return c

def parse_tfrecord(filename_ls):
    filename_queue = tf.train.string_input_producer(filename_ls, shuffle=True)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'c4_': tf.FixedLenFeature([], tf.string),
            'gkm_': tf.FixedLenFeature([], tf.string),
            'dsc_': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )

    c4_1 = tf.reshape(tf.decode_raw(features['c4_'], tf.uint8), [90, 12, 1])
    gkm_1 = tf.reshape(tf.decode_raw(features['gkm_'], tf.uint8), [640, 3, 1])
    dsc_1 = tf.reshape(tf.decode_raw(features['dsc_'], tf.uint8), [90, 39*8, 1])
    label_1 = features['label']
    return c4_1, gkm_1, dsc_1, label_1


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def encode_one_sample(line, ws):
    l = len(line)
    seq_list = []
    for i in range(int(l / ws)):
        seq_list.append(line[i * ws:i * ws + ws])

    
    obj = kmer_featurization(5, 3)
    kmer_features = obj.obtain_frame_sensitive_gapped_kmer_feature_for_a_list_of_sequences(seq_list,
                                                                                           write_number_of_occurrences=True)
    c4_features = sequential_model.obtain_c4_feature_for_a_list_of_sequences(seq_list)
    global COUNT
    dsc_features = dnashape_model.obtain_dsc_feature_for_a_list_of_sequences(COUNT)
    COUNT += 1


    return c4_features, kmer_features.T, dsc_features, int(line[l - 1])


def samples2tfRecord(filename, recordname, ws):
    f = open(filename)
    writer = tf.python_io.TFRecordWriter(recordname)
    for line in f.readlines():
        line = line.strip('\r\n')
        c4, gkm, dsc, label = encode_one_sample(line, ws)

        c4_ = c4.tostring()
        gkm_ = gkm.tostring()
        dsc_ = dsc.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'c4_': _bytes_feature(c4_),
            'gkm_': _bytes_feature(gkm_),
            'dsc_': _bytes_feature(dsc_),
            'label': _int64_feature(label)
        }))
        writer.write(example.SerializeToString())
    writer.close()
    return filename


samples2tfRecord('/mnt/rosetta/cm/20230619-DeepCoding-main_dsc_c4_network/ha/data_ha4.txt', 'data_ha4.tfrecords', 90)

