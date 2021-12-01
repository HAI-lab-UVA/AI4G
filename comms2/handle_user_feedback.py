import pandas as pd
import os

RESPONSES_PATH = os.path.join(os.getcwd(), 'RESPONSES')
if not os.path.isdir(RESPONSES_PATH):
    os.mkdir(RESPONSES_PATH)
    print('RESPONES folder must contain a .csv file.')
    exit()
elif len(os.listdir(RESPONSES_PATH)) == 0:
    print('RESPONES folder must contain a .csv file.')
    exit()
else:
    for file in os.listdir(RESPONSES_PATH):
        os.rename(os.path.join(RESPONSES_PATH, file), os.path.join(RESPONSES_PATH, "Feedback.csv"))
    
# converts responses csv into two dataframes (item pref feedback & allocation pref feedback)
def obtain_feedback_matrices():
    csv_path = os.path.join(RESPONSES_PATH, "Feedback.csv")
    
    feedback = pd.read_csv(csv_path)
    feedback = feedback.drop(columns=['Timestamp'])

    # rename columns in feedback df
    columns = ['User ID']
    for col in feedback.columns:
        if not 'Please' in col:
            if not '1' in col:
                columns.append('alloc_pref_' + col)
            else:
                columns.append('pref_' + col[:-2])
    feedback.columns = columns

    # list of all user IDs (currently based on user in feedback csv)
    user_ids = feedback['User ID']

    # create allocation and item preference dfs
    alloc_feedback = pd.DataFrame(user_ids)
    alloc_feedback = alloc_feedback.join(feedback[columns[1:20]]).fillna(0)
    pref_feedback = pd.DataFrame(user_ids)
    pref_feedback = pref_feedback.join(feedback[columns[20:]]).fillna(0)

    return alloc_feedback, pref_feedback

if __name__ == '__main__':
    alloc_feedback, pref_feedback = obtain_feedback_matrices()

    print("Allocation Feedback Dataframe: ")
    print(alloc_feedback, "\n")
    print("Item Preference Feedback Dataframe: ")
    print(pref_feedback)