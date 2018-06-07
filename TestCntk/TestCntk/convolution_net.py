# Import the relevant modules to be used later
from __future__ import print_function
from mnist_data_loader import load_and_save
from mnist_model_runner import run_model
import sys

import cntk as C
from cntk.learners import sgd, learning_rate_schedule, UnitType
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1)

def create_model(features, num_output_classes, **kwargs):
    with C.layers.default_options(init=C.glorot_uniform(), activation=C.relu):
            h = features
            h = C.layers.Convolution2D(filter_shape=(5,5),
                                       num_filters=8,
                                       strides=(2,2),
                                       pad=True, name='first_conv')(h)
            h = C.layers.Convolution2D(filter_shape=(5,5),
                                       num_filters=16,
                                       strides=(2,2),
                                       pad=True, name='second_conv')(h)
            r = C.layers.Dense(num_output_classes, activation=None, name='classify')(h)
            return r

def Main(argv):
    run_model(create_model, INPUT_DIM_MODEL=(1, 28, 28))

if __name__ == '__main__':
    sys.exit(Main(sys.argv))