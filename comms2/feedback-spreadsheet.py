import numpy as np
import pandas as pd
import os

ALLOCATION_DECISION_PATH = os.path.join(os.getcwd(), 'Allocation_Decision')
FEEDBACK_RESPONSES_PATH = os.path.join(os.getcwd(), 'Feedback_Responses')

if not os.path.isdir(ALLOCATION_DECISION_PATH):
    os.mkdir(ALLOCATION_DECISION_PATH)
    print('Allocation_Decision folder must contain a .csv file.')
    exit()
elif len(os.listdir(ALLOCATION_DECISION_PATH)) == 0:
    print('Allocation_Decision folders must contain a .csv file.')
    exit()
elif not os.path.isdir(FEEDBACK_RESPONSES_PATH):
    os.mkdir(FEEDBACK_RESPONSES_PATH)

def combine_feedback(day):
    allocation_file = None
    for file in os.listdir(ALLOCATION_DECISION_PATH):
        if day in file:
            allocation_file = os.path.join(ALLOCATION_DECISION_PATH, file)
            break
    
    daily_allocation = pd.read_csv(allocation_file)
    daily_allocation.drop('AllocationTime', inplace=True, axis=1)
    user_id = daily_allocation['ID'].tolist()
    daily_allocation.drop("ID", inplace=True, axis=1)
    daily_allocation.columns = [col+day for col in daily_allocation.columns]

    allD = pd.read_csv(os.path.join(os.getcwd(), 'demand_survey.csv'))
    allD.drop("Timestamp", inplace=True, axis=1)
    allD = allD.iloc[:, 1:allD.shape[1]]

    start = (int(day) - 1) * 19
    end = start + 19
    daily_demand = allD.iloc[:, start:end]
    daily_demand.columns = daily_allocation.columns

    feedback = {}

    col1 = []
    for i in range(len(user_id)):
        for j in range(4):
            col1.append(user_id[i])
    feedback["ID"] = col1

    rating = []
    for i in range(len(user_id)):
        rating.append("Requested")
        rating.append("Allocation")
        rating.append("Rate Quantity")
        rating.append("Rate Item")
    feedback["Rating"] = rating

    for item in daily_demand:
        col_list = []
        for j in range(10):
            col_list.append(daily_demand[item].tolist()[j])
            col_list.append(daily_allocation[item].tolist()[j])
            col_list.append("")
            col_list.append("")
        feedback[item] = col_list

    x = pd.DataFrame.from_dict(feedback)
    filename = os.path.join(FEEDBACK_RESPONSES_PATH, "feedback_day" + str(day) + ".csv")
    x.to_csv(filename, index=False)

if __name__ == '__main__':
    selected_day = input('Enter the feedback day to generate spreadsheet: ')
    combine_feedback(selected_day)
    print('Spreadsheet for day ' + selected_day + " has been generated!")