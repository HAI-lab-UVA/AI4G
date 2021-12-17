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

import csv

OVERRIDE = -10000

path = "data\\user_responses.csv"

accuracies = []
save = True
test = False

breakpoint()

if test == True:
    for min_r in [5]:
        for f in [20]:
            for n_e in [0]:
                for ts in [('random',.9),('random',.8),('random',.7),('random',.6),('rows',2),('rows',4),('rows',6),('cols',2),('cols',4),('cols',6)]:
                    print(min_r,f,n_e,ts)
                    acc = 0
                    for ran_num in [5,42,100]:
                        algo = SVDpp_neighborhood(data_path = path, min_ratings = min_r, random_state = ran_num, train_split = ts, n_factors = f, n_epochs = n_e, verbose = False)
                        algo.fit()
                        predictions = algo.test(algo.testset)
                        acc += accuracy.rmse(predictions, verbose=False)
                    accuracies.append([min_r,f,n_e,ts,acc/3])
            if save:
                print("Saving")
                with open("full_full_v2.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerows(accuracies)
                accuracies = []
breakpoint()

algo = SVDpp_neighborhood(data_path = path, n_epochs = 15, verbose = False)


# breakpoint()

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
