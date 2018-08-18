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
            bars = data.DataReader(contract,"iex", start, end)
            return bars
        except:
            retry_cnt += 1
            time.sleep(np.random.randint(1,10))

    print("IEX is not reachable")
    raise Exception('IEX is not reachable')

def download(data_file):
    try:
        data = get_stock_data("SPY", 2013, 1,2,2017,1,1)
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

data_file = os.path.join("data", "Stock", "stock_SPY.pkl")
test_file = os.path.join("test", "Stock", "stock_SPY.pkl")

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
            print("Downloading data from IEX")
            data = download(data_file)
   else:
        data = download(data_file)
    

if __name__ == '__main__':
    sys.exit(Main(sys.argv))