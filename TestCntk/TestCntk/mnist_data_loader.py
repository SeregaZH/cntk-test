import gzip
import numpy as np
import os
import struct
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

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

def load_and_save(num_train_samples, num_test_samples):
    data_dir = os.path.join("..", "Examples", "Image", "DataSets", "MNIST", "data")
    if not os.path.exists(data_dir):
        # URLs for the train image and label data
        train_loader = MnistDataLoader(
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            num_train_samples)

        test_loader = MnistDataLoader(
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
            num_test_samples)

        print("Downloading train data")
        train = train_loader.load()
        
        print("Downloading test data")
        test = test_loader.load()

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

class MnistDataLoader(object):
    def __init__(self, data_url, labels_url, num_samples, *args, **kwargs):
        self.data_url = data_url
        self.labels_url = labels_url
        self.num_samples = num_samples
        self.shape = kwargs['shape'] if 'shape' in kwargs else (28, 28)
        return super().__init__(*args, **kwargs)

    def __loadData(self):
        print ('Downloading ' + self.data_url)
        gzfname, h = urlretrieve(self.data_url, './delete.me')
        print ('Done.')
        try:
            with gzip.open(gzfname) as gz:
                n = struct.unpack('I', gz.read(4))
                # Read magic number.
                if n[0] != 0x3080000:
                    raise Exception('Invalid file: unexpected magic number.')
                # Read number of entries.
                n = struct.unpack('>I', gz.read(4))[0]
                if n != self.num_samples:
                    raise Exception('Invalid file: expected {0} entries.'.format(self.num_samples))
                crow = struct.unpack('>I', gz.read(4))[0]
                ccol = struct.unpack('>I', gz.read(4))[0]
                if crow != self.shape[0] or ccol != self.shape[1]:
                    raise Exception('Invalid file: expected 28 rows/cols per image.')
                # Read data.
                res = np.fromstring(gz.read(self.num_samples * crow * ccol), dtype = np.uint8)
        finally:
            os.remove(gzfname)
        return res.reshape((self.num_samples, crow * ccol))

    def __loadLabels(self):
        print ('Downloading ' + self.labels_url)
        gzfname, h = urlretrieve(self.labels_url, './delete.me')
        print ('Done.')
        try:
            with gzip.open(gzfname) as gz:
                n = struct.unpack('I', gz.read(4))
                # Read magic number.
                if n[0] != 0x1080000:
                    raise Exception('Invalid file: unexpected magic number.')
                # Read number of entries.
                n = struct.unpack('>I', gz.read(4))
                if n[0] != self.num_samples:
                    raise Exception('Invalid file: expected {0} rows.'.format(self.num_samples))
                # Read labels.
                res = np.fromstring(gz.read(self.num_samples), dtype = np.uint8)
        finally:
            os.remove(gzfname)
        return res.reshape((self.num_samples, 1))

    def load(self):
        data = MnistDataLoader.__loadData(self)
        labels = MnistDataLoader.__loadLabels(self)
        return np.hstack((data, labels))