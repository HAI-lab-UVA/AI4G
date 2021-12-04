"""
This module descibes how to test the performances of an algorithm on the
trainset.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from surprise import Dataset, Reader
from surprise import SVD, SVDpp
from surprise import accuracy
from preference import SVDpp_neighborhood, dataset
from surprise.model_selection import KFold
import pandas as pd
import numpy as np

OVERRIDE = -10000

path = "data\\user_responses.csv"

algo = SVDpp_neighborhood(data_path = path, verbose = False)


# Inital ratings matrix R
RHat0 = algo.construct_RHat()
print(RHat0)
# breakpoint()



# Before Training
algo.fit(n_epochs = 0)
RHat1 = algo.construct_RHat(OVERRIDE = OVERRIDE)
print(RHat1)
predictions = algo.test(algo.trainset.build_testset())
accuracy.rmse(predictions, verbose=True)
# breakpoint()

# After Training
algo.fit(n_epochs = 5)
RHat2 = algo.construct_RHat(write_out = False, OVERRIDE = OVERRIDE)
print(RHat2)
predictions = algo.test(algo.trainset.build_testset())
accuracy.rmse(predictions, verbose=True)
# breakpoint()


user_sim = algo.sim_mat(as_df = True, is_user = True)
item_sim = algo.sim_mat(as_df = True, is_user = False)
breakpoint()



# Testing adding a new user
"""
sim_u = algo.sim_mat()
df2 = dataset("data\\user_responses2.csv").iloc[-2]
algo.add_new_user(df2)
RHat3 = algo.construct_RHat(OVERRIDE = OVERRIDE)
print(RHat3)
df2 = dataset("data\\user_responses2.csv").iloc[-1]
algo.add_new_user(df2)
RHat3 = algo.construct_RHat(OVERRIDE = OVERRIDE)
print(RHat3)
breakpoint()
"""
# Handling a new item
