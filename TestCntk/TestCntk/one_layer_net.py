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
    with C.layers.default_options(init = C.glorot_uniform()):
        r = C.layers.Dense(num_output_classes, activation = None)(features)
        return r

def Main(argv):
    run_model(create_model)

if __name__ == '__main__':
    sys.exit(Main(sys.argv))