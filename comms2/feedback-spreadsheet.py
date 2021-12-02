import numpy as np
import pandas as pd

allD = pd.read_csv("C:/Users/student/Documents/Fall21/AI4G/AI4G/learning/results/Predicted Demand.csv")
userID = allD.iloc[:, 0].to_numpy()
allD = allD.iloc[:, 1:allD.shape[1] - 1]


def demand_daily(day, allD):
    start = (day-1) * 19
    end = start + 19
    print(allD.iloc[:, start:end])


demand_daily(2, allD)
demand_daily(3, allD)

def allocation_daily(day):
    allocation_file = "C:/Users/student/Documents/Fall21/AI4G/allocation_day_" + str(day) + ".csv"
    allocation = pd.read_csv(allocation_file)
    allocation.drop('AllocationTime', inplace=True, axis=1)

def combine_feedback(day, allD):
    start = (day - 1) * 19
    end = start + 19
    daily_demand = allD.iloc[:, start:end]
    print(daily_demand)

    allocation_file = "C:/Users/student/Documents/Fall21/AI4G/AI4G/decision/Data/allocation_day " + str(day) + ".csv"
    daily_allocation = pd.read_csv(allocation_file)
    daily_allocation.drop('AllocationTime', inplace=True, axis=1)

    feedback = pd.DataFrame(columns=["User", "Orange", "Apple", "Watermelon", "Banana", "Eggplant", "Tomatoes", "Potatoes", "Bread", "Oats", "Rice", "Fish", "Eggs", "Chicken", "Olive_Oil", "Soy_Bean_Paste", "Beef", "Milk", "Yogurt", "Cheese_Balls"])




combine_feedback(1, allD)



