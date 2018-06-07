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
    with C.layers.default_options(init = C.layers.glorot_uniform(), activation = C.ops.relu):
            h = features
            for _ in range(kwargs.get('num_hidden_layers', 2)):
                h = C.layers.Dense(kwargs.get('hidden_layer_dim', 400))(h)
            r = C.layers.Dense(num_output_classes, activation = None)(h)
            return r

def Main(argv):
    run_model(create_model, num_hidden_layers=3)

if __name__ == '__main__':
    sys.exit(Main(sys.argv))
