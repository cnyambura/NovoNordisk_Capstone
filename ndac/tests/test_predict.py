import numpy as np
import pandas as pd

import matplotlib
%matplotlib inline
#matplotlib.use('agg')
import matplotlib.pyplot as plt

from unittest import mock
from unittest import TestCase
import sys
import io
from io import StringIO, BytesIO
import contextlib

from data_processing import quantile_classify, encode_sequence
from predict import train_clstm


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.model_selection import train_test_split

data = pd.read_csv('/gscratch/pfaendtner/cnyambura/NovoNordisk_Capstone/dataframes/DF_prest.csv', index_col=0)


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.StringIO()
    yield
    sys.stdout = save_stdout


def test_train_clstm():
    
    with nostdout():
        
        df, hist = quantile_classify(data['conc_cf'], data['nt_seq'])
        
        X, y = encode_sequence(df['nt_seq'], df['class'], max_length=200, tag='GACAAGCTTGCGGCCGCA')
        
        x_1, x_test_1, y_1, y_test_1 = train_test_split(X, y, test_size=0.3) 
        
        nt_model_v1 = train_clstm(X, y, test_fraction=0.3, epochs=5, save_file='nt_model_v1.h5')
        
        # check test fraction odd ball input 
        a = 2
        try:
            train_clstm(X, y, test_fraction=a, epochs=2)
        except Exception:
            pass
        else: 
            raise Exception('test fraction must be between 0 and 1. Check function input for error')
    
    #check input data has the same length 
    assert len(X) == len(y), "Check length of input data. X and y data show have the same length."
    
    #check that prediction outputs are between 0 and 1, in addition to being dtype float32
    y_pred_test = nt_model_v1.predict(x_test_1)
    
    assert type(y_pred_test[0][0]) == np.float32, "Check output of model. Prediction values should be dtype np.float64."
    
    assert ((y_pred_test < float(0)).any() == False) & ((y_pred_test > float(1)).any() == False), "Check model activation function. Please ensure that a sigmoid activation function is selected for the dense layer."
    
    sys.stdout.flush()
    sys.stdout = sys.__stdout__
    return

