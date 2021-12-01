# import libraries
import math
import numpy
import pandas as pd

from surprise import SVD, SVDpp
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate


# name definitions
uid_field = "userID"
iid_field = "itemID"
quantity_field = "quantity"
rating_field = "rating"
rating_aggregate_field = "{}_aggregate".format(rating_field)
rating_history_field = "{}_history".format(rating_field)
iid_quantity_field = "iid_quantity" # tuple field containting both item id and quantity

stored_history = 3 # the number of previous ratings stored

class Feedback:
    col_names = [uid_field, iid_field, rating_history_field]
    bp = pd.DataFrame(columns=col_names) # aggregate dataframe
    item_feedback = pd.DataFrame(columns=[uid_field, iid_field, rating_field])
    decision_history = pd.DataFrame(columns=[uid_field, iid_field, "decision_history"])


    def __init__(self):
        self.bp = self.bp.astype({
            uid_field: int,
            iid_field: int,
            # rating_history_field: list
        })


    # helper function to merge ratings dataframes
    def __merge_df(self, df1, df2, merge_on_fields=[uid_field, iid_quantity_field]):
        df = pd.merge(df1, df2, on=merge_on_fields, how="outer")
        df[rating_field] = df["{}_y".format(rating_field)].fillna(df["{}_x".format(rating_field)])
        df.drop(["{}_x".format(rating_field), "{}_y".format(rating_field)], axis=1, inplace=True)
        return df


    def populate_missing_users(self, uid_list, similarity_matrix):
        # use similarity matrix to populate users with no ratings
        # we assume that every user in the aggregate matrix has at least one rating
        pass

    # calculate predictions
    def generate_predictions(self, aggregate_df, decision_df, feedback_df):

        # create SVD model
        # TODO: upgrade to SVD++ and specify parameters
        reader = Reader(rating_scale=(1, 5))

        # Train SVD model from aggregate + similar user data
        data = Dataset.load_from_df(aggregate_df[[uid_field, iid_quantity_field, rating_field]], reader)
        trainset = data.build_full_trainset()
        algo = SVDpp()
        algo.fit(trainset)

        ret_df = decision_df

        ret_df = pd.merge(ret_df, feedback_df, how="left", on=[uid_field, iid_field])

        ret_df[iid_quantity_field] = ret_df[[iid_field, quantity_field]].apply(tuple, axis=1)

        ret_df = pd.merge(ret_df, aggregate_df, how="left", on=[uid_field, iid_quantity_field])
        ret_df[rating_field] = ret_df["{}_y".format(rating_field)].fillna(ret_df["{}_x".format(rating_field)])
        ret_df.drop(["{}_x".format(rating_field), "{}_y".format(rating_field)], axis=1, inplace=True)

        d_id_pairs = set([(uid, iidq) for uid, iidq in ret_df[[uid_field, iid_quantity_field]].values.tolist()])
        a_id_pairs = set([(uid, iidq) for uid, iidq in aggregate_df[[uid_field, iid_quantity_field]].values.tolist()])

        missing_ratings = []
        for id_pairs in d_id_pairs - a_id_pairs:
            uid = id_pairs[0]
            iid_quantity = id_pairs[1]
            rating = algo.predict(str(uid), str(iid_quantity), verbose=True)[3]
            missing_ratings.append([uid, iid_quantity, rating])

        missing_rating_df= pd.DataFrame(missing_ratings, columns=[uid_field, iid_quantity_field, "{}_missing".format(rating_field)])

        ret_df = pd.merge(ret_df, missing_rating_df, how="left", on=[uid_field, iid_quantity_field])
        ret_df[rating_field].fillna(ret_df[rating_field], inplace=True)
        ret_df[rating_field].fillna(ret_df["{}_missing".format(rating_field)], inplace=True)

        return ret_df[[uid_field, iid_field, rating_field]]


    def generate_item_allocation_feedback(self, decision_df, feedback_df):
        allocation_feedback_df = pd.merge(decision_df, feedback_df, how="left", on=[uid_field, iid_field])
        allocation_feedback_df = allocation_feedback_df.dropna(subset=[rating_field])
        allocation_feedback_df[iid_quantity_field] = list(zip(allocation_feedback_df[iid_field], allocation_feedback_df[quantity_field]))
        return allocation_feedback_df[[uid_field, iid_quantity_field, rating_field]]


    # function that runs the entire process
    def predict_feedback(self, decision_df, item_feedback_df, allocation_feedback_df, similarity_data):

        # the easy stuff
        # the step to do right when everything starts
        self.item_feedback = item_feedback_df


        # generate user X (item, allocation) matrix
        item_allocation_feedback_current = self.generate_item_allocation_feedback(decision_df, allocation_feedback_df)

        # make this process more efficient
        # https://stackoverflow.com/questions/48228060/filter-pandas-dataframe-from-tuples
        item_allocation_feedback_history_list = []
        for i, row in self.bp.iterrows():
            user_id = row[uid_field]
            item_id = row[iid_field]
            rating_history = row[rating_history_field]

            decision_history = self.decision_history[(self.decision_history[uid_field] == user_id) & (self.decision_history[iid_field] == item_id)]["decision_history"]

            item_allocation_feedback_history_list.extend([{
                uid_field: user_id,
                iid_quantity_field: (item_id, d),
                rating_field: r
            } for (r, d) in zip(rating_history, decision_history)])

        item_allocation_feedback_history = pd.DataFrame(item_allocation_feedback_history_list, columns=[uid_field, iid_quantity_field, rating_field])

        item_allocation_feedback = pd.concat([item_allocation_feedback_current,item_allocation_feedback_history])
        item_allocation_feedback.drop_duplicates(subset=[uid_field, iid_quantity_field], keep='first', inplace=True)

        # find missing users
        existing_users = set(item_allocation_feedback[uid_field].tolist())
        decision_users = set(decision_df[uid_field].tolist())
        missing_user_ratings = []
        for missing_uid in [uid for uid in decision_users if uid not in existing_users]:

            # sort similarity values from greatest-> least and find similar user that exists in matrix
            for similar_index in numpy.argsort(similarity_data[missing_uid])[::-1]:

                if (similar_index != missing_uid) & (similar_index in existing_users):
                    similar_ratings = item_allocation_feedback[item_allocation_feedback[uid_field] == similar_index].to_dict("records")

                    for similar_rating in similar_ratings:
                        similar_rating[uid_field] = missing_uid
                    missing_user_ratings.extend(similar_ratings)

                    break

        missing_df = pd.DataFrame(missing_user_ratings)

        if not missing_df.empty:
            item_allocation_feedback = pd.concat([item_allocation_feedback, missing_df])

        # generate prediciton matrix
        prediction_df = self.generate_predictions(item_allocation_feedback, decision_df, allocation_feedback_df)

        # add generated to history
        tmp = self.bp
        tmp = pd.merge(self.bp, prediction_df, on=[uid_field, iid_field], how="outer")
        tmp[rating_history_field] = tmp.apply(lambda row: row[rating_history_field].insert(0,row[rating_field]) if type(row[rating_history_field]) == list else [row[rating_field],], axis=1)
        tmp[rating_history_field] = tmp.apply(lambda row: row[rating_history_field].pop() if len(row[rating_history_field]) > stored_history else True, axis=1)
        self.bp = tmp[self.col_names]

        # add decision to history
        tmp = self.decision_history
        tmp = pd.merge(self.decision_history, decision_df, on=[uid_field, iid_field], how="outer")

        tmp["decision_history"] = tmp.apply(lambda row: row["decision_history"].insert(0,row["quantity"]) if type(row["decision_history"]) == list else [row["quantity"],], axis=1)
        tmp["decision_history"] = tmp.apply(lambda row: row["decision_history"].pop() if len(row["decision_history"]) > stored_history else True, axis=1)
        self.decision_history = tmp[[uid_field, iid_field, "decision_history"]]

        return prediction_df

def main():
    print ("feedback_module")

    # inputs to feedback process

    # action "matrix"
    actiondf = pd.DataFrame(
        {'itemID': [0, 2, 0, 2],
        'userID': [0, 0, 1, 1],
        'quantity': [3, 3, 3, 3]}
    )
    print (actiondf)

    # feedback "matrix"
    feedbackdf = pd.DataFrame(
        {'itemID': [0, 2, 0, 2],
        'userID': [0, 0, 1, 1],
        'rating': [3, 3, 3, 3]}
    )
    print (feedbackdf)

    # user similarity "matrix"
    user_similarity = [
    [1, 0.25, 0.5],
    [0.25, 1, 1],
    [0.5, 1, 1],
    ]


    f = Feedback()

    # combine decision and feedback matrices
    decision = pd.DataFrame(
        {'userID': [0, 0, 1, 1, 2, 2],
        'itemID': [0, 1, 0, 1, 0, 1],
        'quantity': [6, 10, 4, 8, 4, 6]}
    )

    feedback = pd.DataFrame(
        {'userID': [0, 1],
        'itemID': [0, 0],
        'rating': [3, 2]}
    )

    print (decision)
    print (feedback)

    x = f.predict_feedback(actiondf, [], feedbackdf, user_similarity)

    print (x)

if __name__ == "__main__":
    main()