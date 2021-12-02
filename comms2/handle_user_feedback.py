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

    # rename columns in feedback df and set user ID as the index
    columns = ['User ID']
    for col in feedback.columns:
        if not 'Please' in col:
            if not '1' in col:
                columns.append('alloc_pref_' + col)
            else:
                columns.append('pref_' + col[:-2])
    feedback.columns = columns
    feedback = feedback.set_index('User ID')
    
    # list of all user IDs
    user_ids = ['Gal Gadot','Anonymous Narwhal','csk6snj','blacklodge','nm2fs','ae3xd','znp3ev','zqz8ae','ak3gj','pex7ps']

    data = []
    for uid in user_ids:
        if uid in feedback.index:
            data.append(feedback.loc[uid].tolist())
        else:
            data.append([])
    feedback = pd.DataFrame(data, columns=columns[1:], index=user_ids).fillna(0)

    # create allocation and item preference dfs
    alloc_feedback = feedback[columns[1:20]]
    pref_feedback = feedback[columns[20:]]

    return alloc_feedback, pref_feedback

if __name__ == '__main__':
    alloc_feedback, pref_feedback = obtain_feedback_matrices()

    print("Allocation Feedback Dataframe: ")
    print(alloc_feedback, "\n")
    print("Item Preference Feedback Dataframe: ")
    print(pref_feedback)