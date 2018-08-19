from __future__ import print_function
import datetime
import sys
import numpy as np
import os
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import cntk as C
import cntk.tests.test_utils
import pickle as  pkl

pd.core.common.is_list_like = pd.api.types.is_list_like
from  pandas_datareader import data

cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components

# Set a random seed
np.random.seed(123)
NUM_DAYS_BACK = 8

def get_stock_data(contract, s_year, s_month, s_day, e_year, e_month, e_day):
    """
    Args:
        contract (str): the name of the stock/etf
        s_year (int): start year for data
        s_month (int): start month
        s_day (int): start day
        e_year (int): end year
        e_month (int): end month
        e_day (int): end day
    Returns:
        Pandas Dataframe: Daily OHLCV bars
    """
    start = datetime.datetime(s_year, s_month, s_day)
    end = datetime.datetime(e_year, e_month, e_day)

    retry_cnt, max_num_retry = 0, 3

    while(retry_cnt < max_num_retry):
        try:
            bars = data.DataReader(contract,'stooq', start, end)
            return bars
        except:
            retry_cnt += 1
            time.sleep(np.random.randint(1,10))

    print("Stooq is not reachable")
    raise Exception('Stooq is not reachable')

def download(data_file):
    try:
        data = get_stock_data("^DJI", 2013, 1,2,2018,1,1)
    except:
        raise Exception("Data could not be downloaded")

    dir = os.path.dirname(data_file)

    if not os.path.exists(dir):
        os.makedirs(dir)

    if not os.path.isfile(data_file):
        print("Saving", data_file )
        with open(data_file, 'wb') as f:
            pkl.dump(data, f, protocol = 2)
    return data

def build_features(data): 
    # Feature name list
    predictor_names = []

    # Compute price difference as a feature
    data["diff"] = np.abs((data["Close"] - data["Close"].shift(1)) / data["Close"]).fillna(0)
    predictor_names.append("diff")

    # Compute the volume difference as a feature
    data["v_diff"] = np.abs((data["Volume"] - data["Volume"].shift(1)) / data["Volume"]).fillna(0)
    predictor_names.append("v_diff")

    for i in range(1, NUM_DAYS_BACK + 1):
        data["p_" + str(i)] = np.where(data["Close"] > data["Close"].shift(i), 1, 0) # i: number of look back days
        predictor_names.append("p_" + str(i))
    
    data["next_day"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)
    data["next_day_opposite"] = np.where(data["next_day"]==1, 0, 1)
    train_data = data["2015-01-20":"1990-02-05"]
    train_test = data["2018-01-20":"2015-01-25"]

    # If you want to save the file to your local drive
    data.to_csv("f_params.csv")
    return predictor_names, train_data, train_test

data_file = os.path.join("data", "Stock", "stock_DJI.pkl")
test_file = os.path.join("test", "Stock", "stock_DJI.pkl")

def Main(argv):
   # Check for data in local cache
   argv[0] = 'test'
   if  len(argv) > 0 and argv[0] == 'test':
       if os.path.isfile(test_file): 
            print("Reading data from test data directory")
            data = pd.read_pickle(test_file)
       else:
         if os.path.exists(data_file):
            print("File already exists", data_file)
            data = pd.read_pickle(data_file)
         else:
            print("Test data directory missing file", test_file)
            print("Downloading data from stooq")
            data = download(data_file)
   else:
        data = download(data_file)
   features_names, training_data, test_data = build_features(data);
   training_features = np.asarray(training_data[predictor_names], dtype = "float32")
   training_labels = np.asarray(training_data[["next_day","next_day_opposite"]], dtype="float32")

if __name__ == '__main__':
    sys.exit(Main(sys.argv))