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


def dump_libsvm_file(X, y, file, num_feature_names, cat_feature_names, n_categories, use_field=False, decimals=6,
                     use_hash=False, n_bins=1000000):
    feature_to_index = _get_feature_to_index(num_feature_names, cat_feature_names, n_categories, use_field)
    with open(file, 'w') as f:
        for i, row in X.iterrows():
            parsed_row = str(y.loc[i]) if y is not None else ''
            for feature in num_feature_names:
                index = str(feature_to_index[feature])
                field = ''.join([index, ':']) if use_field else ''
                parsed_row = ''.join(
                    [parsed_row, ' ', index, ':', field, str(round(row[feature], decimals))])
            for feature in cat_feature_names:
                if use_field:
                    field = feature_to_index[feature]
                    index = int(row[feature])
                    index = _hash_str(str(index), n_bins) if use_hash and n_categories[feature] > n_bins else index
                    parsed_row = ''.join([parsed_row, ' ', str(field), ':', str(index), ':1'])
                else:
                    index = int(row[feature])
                    index = _hash_str(str(index), n_bins) if use_hash and n_categories[feature] > n_bins else index
                    index = feature_to_index['_'.join([feature, str(index)])]
                    parsed_row = ''.join([parsed_row, ' ', str(index), ':1'])
            f.write(parsed_row.lstrip() + '\n')


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _serialize_example(feature):
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def dump_tfrecord_file(X, y, file, num_feature_names, cat_feature_names, target_name=None, decimals=6):
    options = tf.io.TFRecordOptions(compression_type='GZIP')
    with tf.io.TFRecordWriter(path=file, options=options) as writer:
        parsed_row = dict()
        for i, row in X.iterrows():
            if y is not None:
                parsed_row[target_name] = _int64_feature(y.loc[i])
            for feature in num_feature_names:
                parsed_row[feature] = _float_feature(round(row[feature], decimals))
            for feature in cat_feature_names:
                parsed_row[feature] = _int64_feature(int(row[feature]))
            writer.write(_serialize_example(parsed_row))
