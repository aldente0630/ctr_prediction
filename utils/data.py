import hashlib
import tensorflow as tf


def _hash_str(string, n_bins):
    return int(hashlib.md5(string.encode('utf8')).hexdigest(), 16) % (n_bins - 1) + 1


def _get_feature_to_index(num_feature_names, cat_feature_names, n_categories, use_field):
    if use_field:
        feature_to_index = {feature: i for i, feature in enumerate(num_feature_names | cat_feature_names)}
    else:
        feature_to_index = {feature: i for i, feature in enumerate(num_feature_names)}
        j = 0
        for feature in cat_feature_names:
            for label in range(n_categories[feature]):
                feature_to_index['_'.join([feature, str(label)])] = len(num_feature_names) + j
                j += 1
    return feature_to_index


def dump_libsvm_file(X, y, file, num_feature_names, cat_feature_names, n_categories, use_field=False, decimals=8,
                     use_hash=False, n_bins=1000000):
    feature_to_index = _get_feature_to_index(num_feature_names, cat_feature_names, n_categories, use_field)
    with open(file, 'w') as f:
        for i, row in X.iterrows():
            if y is not None:
                serialized_row = str(y.loc[i])
            else:
                serialized_row = ''
            for feature in num_feature_names:
                index = str(feature_to_index[feature])
                field = ''.join([index, ':']) if use_field else ''
                serialized_row = ''.join(
                    [serialized_row, ' ', index, ':', field, str(round(row[feature], decimals))])
            for feature in cat_feature_names:
                if use_field:
                    field = feature_to_index[feature]
                    index = int(row[feature])
                    index = _hash_str(str(index), n_bins) if use_hash and n_categories[feature] > n_bins else index
                    serialized_row = ''.join([serialized_row, ' ', str(field), ':', str(index), ':1'])
                else:
                    index = int(row[feature])
                    index = _hash_str(str(index), n_bins) if use_hash and n_categories[feature] > n_bins else index
                    index = feature_to_index['_'.join([feature, str(index)])]
                    serialized_row = ''.join([serialized_row, ' ', str(index), ':1'])
            f.write(serialized_row.lstrip() + '\n')


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _serialize_example(feature):
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def dump_tfrecord_file(X, y, file, num_feature_names, cat_feature_names, target_name=None, key_names=None,
                       decimals=8, compression_type=None):
    options = tf.io.TFRecordOptions(compression_type=compression_type)
    with tf.io.TFRecordWriter(path=file, options=options) as writer:
        serialized_row = dict()
        for i, row in X.iterrows():
            if key_names is not None:
                for key_name in key_names:
                    serialized_row[key_name] = _bytes_feature(row[key_name])
            if y is not None:
                serialized_row[target_name] = _int64_feature(y.loc[i])
            for feature in num_feature_names:
                serialized_row[feature] = _float_feature(round(row[feature], decimals))
            for feature in cat_feature_names:
                serialized_row[feature] = _int64_feature(int(row[feature]))
            writer.write(_serialize_example(serialized_row))
