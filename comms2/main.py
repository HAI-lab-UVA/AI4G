import argparse
from numpy import genfromtxt
import pandas as pd

from feedback_module import Feedback


def main():

    parser = argparse.ArgumentParser(description='Feedback module')
    parser.add_argument('decision', help='Decision Matrix data in csv form')
    parser.add_argument('alloc', help='Allocation Feedback data in csv form')
    parser.add_argument('item', help='Item Feedback data in csv form')
    parser.add_argument('usersim', help='User similarity data in csv form')
    parser.add_argument('itemsim', help='Item similarity data in csv form')
    parser.add_argument('-o', '--outfile', default="predicted_feedback",  help='output csv filename')

    args = parser.parse_args()

    # clean this part up
    user_mapping_file = "../comms-1/data/user_responses.csv"
    user_data = pd.read_csv(user_mapping_file)
    user_id_mapping = {item: i for i, item in user_data["Please enter an identifier (ex. computing id)"].items()}
    id_user_mapping = {v: k for k, v in user_id_mapping.items()}

    item_mapping_file = "../comms-1/data/item_list.csv"
    item_data = pd.read_csv(item_mapping_file)
    item_id_mapping = {item: i for i, item in item_data["Item_name"].items()}
    id_item_mapping = {v: k for k, v in item_id_mapping.items()}

    decision_df = pd.read_csv(args.decision)

    decision_list = []
    for i, row in decision_df.iterrows():
        user_name = row["ID"]

        decision_list.extend([{
            'userID': user_id_mapping[user_name],
            'itemID': int(item_id),
            'quantity': quantity,
        } for item_id, quantity in row[[str(i) for i in range(18)]].items()])
    
    decision_df = pd.DataFrame(decision_list)

    alloc_feedback_df = pd.read_csv(args.alloc)
    item_feedback_df = pd.read_csv(args.item)


    alloc_feedback_list = []
    for i, row in alloc_feedback_df.iterrows():
        user_name = row[0]

        alloc_feedback_list.extend([{
            'userID': user_id_mapping[user_name],
            'itemID': item_id,
            'rating': rating,
        } for item_id, (item_name, rating) in enumerate(row[range(1,18)].items()) if rating > 0])
    
    alloc_feedback_df = pd.DataFrame(alloc_feedback_list)

    item_feedback_list = []
    for i, row in item_feedback_df.iterrows():
        user_name = row[0]

        item_feedback_list.extend([{
            'userID': user_id_mapping[user_name],
            'itemID': item_id,
            'rating': rating,
        } for item_id, (item_name, rating) in enumerate(row[range(1,18)].items()) if rating > 0])
    
    item_feedback_df = pd.DataFrame(item_feedback_list)

    usersim_df = genfromtxt(args.usersim, delimiter=',')
    itemsim_df = genfromtxt(args.itemsim, delimiter=',')

    f = Feedback()

    feedback_prediction_df = f.predict_feedback(decision_df, item_feedback_df, alloc_feedback_df, usersim_df, itemsim_df)

    feedback_prediction_df['username'] = feedback_prediction_df.apply(lambda x: id_user_mapping[x['userID']], axis=1)
    feedback_prediction_df['item'] = feedback_prediction_df.apply(lambda x: id_item_mapping[x['itemID']], axis=1)
    feedback_prediction_df[['userID', 'itemID', 'username', 'item', 'rating']].to_csv("{}.csv".format(args.outfile), index=False)


if __name__ == "__main__":
    main()