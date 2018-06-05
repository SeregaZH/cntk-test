# Import the relevant modules to be used later
from __future__ import print_function
import gzip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import struct
import sys
import math

import cntk as C
from cntk.learners import sgd, learning_rate_schedule, UnitType
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1)

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

# Functions to load MNIST images and unpack into train and test set.
# - loadData reads a image and formats it into a 28x28 long array
# - loadLabels reads the corresponding label data, one for each image
# - load packs the downloaded image and label data into a combined format to be read later by
#   the CNTK text reader

def loadData(src, cimg):
    print ('Downloading ' + src)
    gzfname, h = urlretrieve(src, './delete.me')
    print ('Done.')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number.
            if n[0] != 0x3080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))[0]
            if n != cimg:
                raise Exception('Invalid file: expected {0} entries.'.format(cimg))
            crow = struct.unpack('>I', gz.read(4))[0]
            ccol = struct.unpack('>I', gz.read(4))[0]
            if crow != 28 or ccol != 28:
                raise Exception('Invalid file: expected 28 rows/cols per image.')
            # Read data.
            res = np.fromstring(gz.read(cimg * crow * ccol), dtype = np.uint8)
    finally:
        os.remove(gzfname)
    return res.reshape((cimg, crow * ccol))

def loadLabels(src, cimg):
    print ('Downloading ' + src)
    gzfname, h = urlretrieve(src, './delete.me')
    print ('Done.')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number.
            if n[0] != 0x1080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))
            if n[0] != cimg:
                raise Exception('Invalid file: expected {0} rows.'.format(cimg))
            # Read labels.
            res = np.fromstring(gz.read(cimg), dtype = np.uint8)
    finally:
        os.remove(gzfname)
    return res.reshape((cimg, 1))

def try_download(dataSrc, labelsSrc, cimg):
    data = loadData(dataSrc, cimg)
    labels = loadLabels(labelsSrc, cimg)
    return np.hstack((data, labels))

# Save the data files into a format compatible with CNTK text reader
def savetxt(filename, ndarray):
    dir = os.path.dirname(filename)

    if not os.path.exists(dir):
        os.makedirs(dir)

    if not os.path.isfile(filename):
        print("Saving", filename )
        with open(filename, 'w') as f:
            labels = list(map(' '.join, np.eye(10, dtype=np.uint).astype(str)))
            for row in ndarray:
                row_str = row.astype(str)
                label_str = labels[row[-1]]
                feature_str = ' '.join(row_str[:-1])
                f.write('|labels {} |features {}\n'.format(label_str, feature_str))
    else:
        print("File already exists", filename)

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

def create_model(features, num_output_classes):
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

def load_data(num_train_samples, num_test_samples):
    data_dir = os.path.join("..", "Examples", "Image", "DataSets", "MNIST", "data")
    if not os.path.exists(data_dir):
        # URLs for the train image and label data
        url_train_image = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
        url_train_labels = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'

        print("Downloading train data")
        train = try_download(url_train_image, url_train_labels, num_train_samples)

        # URLs for the test image and label data
        url_test_image = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
        url_test_labels = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

        print("Downloading test data")
        test = try_download(url_test_image, url_test_labels, num_test_samples)

        # Save the train and test files (prefer our default path for the data)
        
        if not os.path.exists(data_dir):
            data_dir = os.path.join("..", "Examples", "Image", "DataSets", "MNIST", "data")

        print ('Writing train text file...')
        savetxt(os.path.join(data_dir, "Train-28x28_cntk_text.txt"), train)

        print ('Writing test text file...')
        savetxt(os.path.join(data_dir, "Test-28x28_cntk_text.txt"), test)

        print('Examples loades')

    train_file = os.path.join(data_dir, "Train-28x28_cntk_text.txt")
    test_file = os.path.join(data_dir, "Test-28x28_cntk_text.txt")
    if not os.path.isfile(train_file) or not os.path.isfile(test_file):
        raise ValueError("Please generate the data by completing CNTK 103 Part A")

    print("Data directory is {0}".format(data_dir))
    return train_file, test_file

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

def Main(argv):

    NUM_TRAIN_SAMPLES = 60000
    NUM_TEST_SAMPLES = 10000
    INPUT_DIM_MODEL = (1, 28, 28)
    INPUT_DIM = 28*28
    NUM_OUTPUT_CLASSES = 10
    LEARNING_RATE = 0.2
    MINIBATCH_SIZE = 64
    NUM_SAMPLES_PER_SWEEP = 60000
    NUM_SWEEP_TO_TRAIN = 5

    train_file, test_file = load_data(NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES)

    input = C.input_variable(INPUT_DIM_MODEL)
    label = C.input_variable(NUM_OUTPUT_CLASSES)
    z = create_model(input/255.0, NUM_OUTPUT_CLASSES)
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

if __name__ == '__main__':
    sys.exit(Main(sys.argv))