import numpy as np
import pandas as pd

def combine_feedback(day):
    allD = pd.read_csv("C:/Users/student/Documents/Fall21/AI4G/AI4G/learning/results/Predicted Demand.csv")
    allD = allD.iloc[:, 1:allD.shape[1] - 1]

    start = (day - 1) * 19
    end = start + 19
    daily_demand = allD.iloc[:, start:end]

    allocation_file = "C:/Users/student/Documents/Fall21/AI4G/AI4G/decision/Data/allocation_day " + str(day) + ".csv"
    daily_allocation = pd.read_csv(allocation_file)
    daily_allocation.drop('AllocationTime', inplace=True, axis=1)
    user_id = daily_allocation['ID'].tolist()
    daily_allocation.drop("ID", inplace=True, axis=1)
    daily_allocation.columns = list(daily_demand.columns)

    feedback = {}

    col1 = []
    for i in range(len(user_id)-1):
        for j in range(4):
            col1.append(user_id[i])
    feedback["ID"] = col1

    rating = []
    for i in range(len(user_id)-1):
        rating.append("Requested")
        rating.append("Allocation")
        rating.append("Rate Quantity")
        rating.append("Rate Item")
    feedback["Rating"] = rating

    for item in daily_demand:
        col_list = []
        for j in range(9):
            col_list.append(daily_demand[item].tolist()[j])
            col_list.append(daily_allocation[item].tolist()[j])
            col_list.append("")
            col_list.append("")
        feedback[item] = col_list

    x = pd.DataFrame.from_dict(feedback)
    filename = "feedback_day" + str(day) + ".csv"
    x.to_csv(filename, index=False)


combine_feedback(1)
