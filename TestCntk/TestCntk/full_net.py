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
    with cntk.layers.default_options(activation=cntk.ops.relu, pad=False):
        return cntk.layers.Sequential([
            cntk.layers.Convolution2D((5,5), num_filters=32, pad=True),
            cntk.layers.MaxPooling((3,3), strides=(2,2)),
            cntk.layers.Convolution2D((3,3), num_filters=48),
            cntk.layers.MaxPooling((3,3), strides=(2,2)),
            cntk.layers.Convolution2D((3,3), num_filters=64),
            cntk.layers.Dense(96),
            cntk.layers.Dropout(dropout_rate=0.5),
            cntk.layers.Dense(num_output_classes, activation=None) # no activation in final layer (softmax is done in criterion)
        ])(features)

def Main(argv):
    run_model(create_model, INPUT_DIM_MODEL=(1, 28, 28))

if __name__ == '__main__':
    sys.exit(Main(sys.argv))