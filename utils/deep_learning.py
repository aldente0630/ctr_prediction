import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib


def _from_tfrecord(serialized, features, target_name, key_names):
    example = tf.io.parse_single_example(serialized=serialized, features=features)
    if key_names is not None:
        for key_name in key_names:
            _ = example.pop(key_name, None)
    if target_name is not None:
        target = example.pop(target_name, None)
        return example, target
    else:
        return example


def extract_dataset(file_paths, compression_type=None, shuffle_buffer_size=1024, is_training=True):
    files = tf.data.Dataset.list_files(file_paths, shuffle=False)
    if is_training:
        dataset = files.interleave(
            lambda file: tf.data.TFRecordDataset(file, compression_type=compression_type),
            num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(shuffle_buffer_size, seed=42)
    else:
        dataset = tf.data.TFRecordDataset(files, compression_type=compression_type)
    return dataset


def transform_dataset(dataset, num_feature_names, cat_feature_names, target_name=None, key_names=None):
    features = dict()
    if key_names is not None:
        for key_name in key_names:
            features[key_name] = tf.io.FixedLenFeature([], tf.string)
    if target_name is not None:
        features[target_name] = tf.io.FixedLenFeature([], tf.int64)
    for feature in num_feature_names:
        features[feature] = tf.io.FixedLenFeature([], tf.float32)
    for feature in cat_feature_names:
        features[feature] = tf.io.FixedLenFeature([], tf.int64)
    dataset = dataset.map(lambda serialized: _from_tfrecord(
        serialized=serialized, features=features, target_name=target_name, key_names=key_names),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def load_dataset(dataset, batch_size=32, is_training=True):
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    if is_training:
        return dataset.repeat()
    else:
        return dataset


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_n_examples(dataset):
    n_examples = 0
    for batch in dataset.take(-1):
        shape = list(batch[0].values())[0].shape
        if len(shape) == 0:
            n_examples += 1
        else:
            n_examples += shape[0]
    return n_examples


def get_n_steps(total_size, batch_size):
    n_steps = total_size // batch_size
    if total_size % batch_size > 0:
        n_steps += 1
    return int(n_steps)


# This function is not yet checked whether it is thread-safe or not.
def get_target(dataset):
    target = np.array([])
    for batch in dataset.take(-1):
        target = np.concatenate([target, batch[1].numpy()])
    return target
