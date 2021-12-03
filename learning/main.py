import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
import numpy as np


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation, Dropout
import time

from load_data import *

print("all libs loaded successfully...")

def load_data():
    S = load_supply()
    D = load_demand()
    P = load_priority()
    A_prev = load_allocation_at_previous_step()
    A_history = load_allocation_history()
    
    n_S = get_number_of_suppliers()
    n_U = get_number_of_users()
