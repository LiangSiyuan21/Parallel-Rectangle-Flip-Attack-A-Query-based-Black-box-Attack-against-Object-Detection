"""
Helper functions
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import json
import os
import sys

import numpy as np
import psutil
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from lib.challenges.cifar10_challenge.model import Model as cifar10_model
from lib.challenges.imagenet_challenge.model import Model as imagenet_model
from lib.challenges.mnist_challenge.model import Model as mnist_model


def load_config(config_file):
    """
    Loads configuration as a json from a file `config_file` located in
    `config-jsons`
    :param config_file:
    :return:
    """
    config_file = config_path_join(config_file)
    with open(config_file) as config_file:
        config = json.load(config_file)

    return config


def construct_attack(model, config, dset):
    """
    Construct attack for `model`
    and configurations `config`
    and dataset `dset`
    :param model:
    :param config:
    :param dset:
    :return:
    """
    return eval(config['attack_name'])(model,
                                       **config['attack_config'],
                                       lb=dset.min_value,
                                       ub=dset.max_value)


def check_shape(np_arr, target_shape):
    """
    check the shape of `np_arr` against the target shape `target_shape`
    :param np_arr:
    :param target_shape:
    :return:
    """
    if np_arr.shape != target_shape:
        raise Exception('Invalid shape: expected ({}), found {}'.format(target_shape, np_arr.shape))


def check_values(np_arr, min_val, max_val):
    if np.amax(np_arr) > max_val or \
            np.amin(np_arr) < min_val or \
            np.isnan(np.amax(np_arr)):
        raise Exception('Invalid pixel range. Expected [{}, {}], found [{}, {}]'.format(
            min_val,
            max_val,
            np.amin(np_arr),
            np.amax(np_arr)))


def get_model_file(config):
    """
    Returns the checkpoint folder or model file
    :param config:
    :return:
    """
    if 'model_dir' in config:
        model_file = tf.train.latest_checkpoint(data_path_join(config['model_dir']))
    else:
        model_file = data_path_join(config['model_file'])
    if model_file is None:
        raise Exception('No checkpoint found')
    return model_file


def construct_model(model_name):
    """
    Load the corresponding model
    :param dset_name:
    :return:
    """
    if model_name == 'cifar10':
        return cifar10_model()
    elif model_name == 'mnist':
        return mnist_model()
    elif model_name == 'imagenet':
        return imagenet_model()
    else:
        raise Exception('Unknown dataset.')


def get_dataset_shape(dset_name):
    if dset_name == 'mnist':
        dset_shape = (784,)
    elif dset_name == 'cifar10':
        dset_shape = (32, 32, 3)
    elif dset_name == 'imagenet':
        dset_shape = (299, 299, 3)
    elif dset_name == 'image_sub':
        dset_shape = (299, 299, 3)
    elif dset_name == 'tiny':
        dset_shape = (64, 64, 3)
    else:
        raise Exception('Unsupported dataset for attack yet')

    return dset_shape


def memory_usage():
    """
    return the memory usage in GB
    """
    p = psutil.Process(os.getpid())
    mem = p.memory_full_info().uss / float(2 ** 30)
    print("Memory Usage in GB: {}".format(mem))


def free_gpus():
    os.system(r"""
    nvidia-smi | sed -n 's/|\s*[0-9]*\s*\([0-9]*\)\s*.*python.*/\1/p' | sort | uniq| parallel 'echo {} uses GPU; kill -9 {}'; echo 'GPUs freed'""")


def get_files(_dir, ext):
    """
    get all the files in `_dir` with extension `ext
    :param _dir:
    :param ext:
    :return:
    """
    return glob.glob(os.path.join(_dir, '*.' + ext))


def create_dir(_dir):
    """Create a directory, skip if it exists"""
    os.makedirs(_dir, exist_ok=True)


def get_plt_dir():
    """returs the {REPO_PATH}/plots"""
    _dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(_dir[:_dir.rfind('src')], 'plots')


def get_src_dir():
    """returns the {REPO_PATH}/src/"""
    _dir = os.path.dirname(os.path.abspath(__file__))
    return _dir[:_dir.rfind('src') + 3]


def get_config_dir():
    """returs the {REPO_PATH}/d"""
    _dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(get_src_dir(), 'config-jsons')


def get_data_dir():
    """returs the {REPO_PATH}/data"""
    _dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(_dir[:_dir.rfind('src')], 'data')


def plt_path_join(*kwargs):
    """
    reutrns path to the file whose dir information are provided in kwargs
    similar to `os.path.join`
    :param kwargs:
    :return:
    """
    return os.path.join(get_plt_dir(), *kwargs)


def src_path_join(*kwargs):
    """
    reutrns path to the file whose dir information are provided in kwargs
    similar to `os.path.join`
    :param kwargs:
    :return:
    """
    return os.path.join(get_src_dir(), *kwargs)


def data_path_join(*kwargs):
    """
    reutrns path to the file whose dir information are provided in kwargs
    similar to `os.path.join`
    :param kwargs:
    :return:
    """
    return os.path.join(get_data_dir(), *kwargs)


def config_path_join(filename):
    """
    returns abs pathname to the config of name `filename`
    assuming it is in `config-jsons`
    :param filename:
    :return:
    """
    return src_path_join('config-jsons', filename)
