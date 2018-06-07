# Import the relevant modules to be used later
from __future__ import print_function
from mnist_data_loader import load_and_save
import gzip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import numpy as np
import shutil
import struct
import math

import cntk as C
from cntk.learners import sgd, learning_rate_schedule, UnitType
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1)

# Read a CTF formatted text (as mentioned above) using the CTF deserializer from a file
def create_reader(path, is_training, input_dim, num_label_classes):

    labelStream = C.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False)
    featureStream = C.io.StreamDef(field='features', shape=input_dim, is_sparse=False)

    deserailizer = C.io.CTFDeserializer(path, C.io.StreamDefs(labels = labelStream, features = featureStream))

    return C.io.MinibatchSource(deserailizer,
       randomize = is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)

# Define a utility function to compute the moving average sum.
# A more efficient implementation is possible with np.cumsum() function
def moving_average(a, w=5):
    if len(a) < w:
        return a[:]    # Need to send a copy of the array
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

def plot_learning(plotdata): 
    # Compute the moving average loss to smooth out the noise in SGD
    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plotdata["avgerror"] = moving_average(plotdata["error"])

    # Plot the training loss and the training error
    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')

    plt.show()

    plt.subplot(212)
    plt.plot(plotdata["batchsize"], plotdata["avgerror"], 'r--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Label Prediction Error')
    plt.title('Minibatch run vs. Label Prediction Error')
    plt.show()

def test_model(test_map, reader, trainer, num_samples, test_minibatch_size):

    # Test data for trained model
    num_minibatches_to_test = num_samples // test_minibatch_size
    test_result = 0.0

    for i in range(num_minibatches_to_test):

        # We are loading test data in batches specified by test_minibatch_size
        # Each data point in the minibatch is a MNIST digit image of 784 dimensions
        # with one pixel per dimension that we will encode / decode with the
        # trained model.
        data = reader.next_minibatch(test_minibatch_size, input_map = test_map)

        eval_error = trainer.test_minibatch(data)
        test_result = test_result + eval_error

    # Average of evaluation errors of all test minibatches
    print("Average test error: {0:.2f}%".format(test_result*100 / num_minibatches_to_test))

class Trainer:
    
    def __init__(self, minibatch_size, samples_per_sweep, sweep_train, trainer, reader_train):
        self.minibatch_size = minibatch_size
        self.training_progress_output_freq = 500
        self.num_minibatches_to_train = math.floor((samples_per_sweep * sweep_train) / minibatch_size)
        self.trainer = trainer
        self.reader_train = reader_train
    
    def train(self, input_map):
        # Run the trainer on and perform model training
        plotdata = {"batchsize":[], "loss":[], "error":[]}

        for i in range(0, int(self.num_minibatches_to_train)):

            # Read a mini batch from the training data file
            data = self.reader_train.next_minibatch(self.minibatch_size, input_map = input_map)

            self.trainer.train_minibatch(data)
            batchsize, loss, error = self.print_training_progress(i, self.training_progress_output_freq, verbose=1)

            if not (loss == "NA" or error =="NA"):
                plotdata["batchsize"].append(batchsize)
                plotdata["loss"].append(loss)
                plotdata["error"].append(error)

        return plotdata

    # Defines a utility that prints the training progress
    def print_training_progress(self, mb, frequency, verbose=1):
        training_loss = "NA"
        eval_error = "NA"

        if mb%frequency == 0:
            training_loss = self.trainer.previous_minibatch_loss_average
            eval_error = self.trainer.previous_minibatch_evaluation_average
            if verbose:
                print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error*100))

        return mb, training_loss, eval_error

def run_model(create_model_fn, **params):
    NUM_TRAIN_SAMPLES = params.get('NUM_TRAIN_SAMPLES', 60000)
    NUM_TEST_SAMPLES = params.get('NUM_TEST_SAMPLES', 10000)
    INPUT_DIM_MODEL = params.get('INPUT_DIM_MODEL', 28*28)
    INPUT_DIM = params.get('INPUT_DIM', 28*28)
    NUM_OUTPUT_CLASSES = params.get('NUM_OUTPUT_CLASSES', 10)
    LEARNING_RATE = params.get('LEARNING_RATE', 0.2)
    MINIBATCH_SIZE = params.get('MINIBATCH_SIZE', 64)
    NUM_SAMPLES_PER_SWEEP = params.get('NUM_SAMPLES_PER_SWEEP', 60000)
    NUM_SWEEP_TO_TRAIN = params.get('NUM_SWEEP_TO_TRAIN', 10)

    train_file, test_file = load_and_save(NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES)

    input = C.input_variable(INPUT_DIM_MODEL)
    label = C.input_variable(NUM_OUTPUT_CLASSES)
    z = create_model_fn(input/255.0, NUM_OUTPUT_CLASSES, **params)
    loss = C.cross_entropy_with_softmax(z, label)
    label_error = C.classification_error(z, label)
    lr_schedule = learning_rate_schedule(LEARNING_RATE, UnitType.minibatch)
    learner = sgd(z.parameters, lr_schedule)    
    trainer = C.Trainer(z, (loss, label_error), [learner])

    # Create the reader to training data set
    reader_train = create_reader(train_file, True, INPUT_DIM, NUM_OUTPUT_CLASSES)

    # Map the data streams to the input and labels.
    input_map = {
        label  : reader_train.streams.labels,
        input  : reader_train.streams.features
    }

    tr = Trainer(MINIBATCH_SIZE, NUM_SAMPLES_PER_SWEEP, NUM_SWEEP_TO_TRAIN, trainer, reader_train)
    plotdata = tr.train(input_map)
    plot_learning(plotdata)

    # Read the training data
    reader_test = create_reader(test_file, False, INPUT_DIM, NUM_OUTPUT_CLASSES)

    test_input_map = {
        label  : reader_test.streams.labels,
        input  : reader_test.streams.features,
    }

    test_model(test_input_map, reader_test, trainer, NUM_TEST_SAMPLES, 512)
    return z
