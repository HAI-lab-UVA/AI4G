import pandas as pd
import os

FEEDBACK_RESPONSES_PATH = os.path.join(os.getcwd(), 'Feedback_Responses')
if not os.path.isdir(FEEDBACK_RESPONSES_PATH):
    os.mkdir(FEEDBACK_RESPONSES_PATH)
    print('Feedback_Responses folder must contain a .csv file.')
    exit()
elif len(os.listdir(FEEDBACK_RESPONSES_PATH)) == 0:
    print('Feedback_Responses folder must contain a .csv file.')
    exit()

SEPARATE_RESPONSES_PATH = os.path.join(os.getcwd(), 'Separate_Responses')
if not os.path.isdir(SEPARATE_RESPONSES_PATH):
    os.mkdir(SEPARATE_RESPONSES_PATH)

def obtain_feedback_matrices(day):
    csv_path = None
    for file in os.listdir(FEEDBACK_RESPONSES_PATH):
        if day in file:
            csv_path = os.path.join(FEEDBACK_RESPONSES_PATH, file)
            break

    feedback = pd.read_csv(csv_path)
    columns = feedback.columns

    # list of all user IDs
    user_ids = ['Gal Gadot','Anonymous Narwhal','csk6snj','blacklodge','nm2fs','ae3xd','znp3ev','zqz8ae','ak3gj','pex7ps']

    alloc_feedback_data = []
    pref_feedback_data = []
    for uid in user_ids:
        quantity_row_exists = ((feedback['Rating']=='Rate Quantity') & (feedback['ID']==uid)).any()
        item_row_exists = ((feedback['Rating']=='Rate Item') & (feedback['ID']==uid)).any()
        
        if quantity_row_exists:
            row_index = feedback.index[(feedback['Rating']=='Rate Quantity') & (feedback['ID']==uid)].tolist()[0]
            alloc_feedback_data.append(feedback.loc[row_index, columns[2:]].tolist())
        else:
            alloc_feedback_data.append([])

        if item_row_exists:
            row_index = feedback.index[(feedback['Rating']=='Rate Item') & (feedback['ID']==uid)].tolist()[0]
            pref_feedback_data.append(feedback.loc[row_index, columns[2:]].tolist())
        else:
            pref_feedback_data.append([])

    alloc_feedback = pd.DataFrame(alloc_feedback_data, columns=columns[2:], index=user_ids).fillna(0)
    pref_feedback = pd.DataFrame(pref_feedback_data, columns=columns[2:], index=user_ids).fillna(0)

    return alloc_feedback, pref_feedback

if __name__ == '__main__':
    selected_day = input('Enter the feedback day: ')
    alloc_feedback, pref_feedback = obtain_feedback_matrices(selected_day)

    # print("Allocation Feedback Dataframe: ")
    # print(alloc_feedback, "\n")
    # print("Item Preference Feedback Dataframe: ")
    # print(pref_feedback)

    alloc_feedback.to_csv(os.path.join(SEPARATE_RESPONSES_PATH, str('allocation_feedback_day' + selected_day + '.csv')))
    pref_feedback.to_csv(os.path.join(SEPARATE_RESPONSES_PATH, str('pref_feedback_day' + selected_day + '.csv')))
    print('Allocation and Preference Feedback spreadsheets have been generated for day ' + selected_day)


    
        
    

