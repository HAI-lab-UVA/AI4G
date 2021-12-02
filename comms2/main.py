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

    decision_df = pd.read_csv(args.decision)

    alloc_feedback_df = pd.read_csv(args.alloc)
    item_feedback_df = pd.read_csv(args.item)

    usersim_df = genfromtxt(args.usersim, delimiter=',')
    itemsim_df = genfromtxt(args.itemsim, delimiter=',')

    f = Feedback()

    feedback_prediction_df = f.predict_feedback(decision_df, item_feedback_df, alloc_feedback_df, usersim_df, itemsim_df)

    feedback_prediction_df.to_csv("{}.csv".format(args.outfile), index=False)


if __name__ == "__main__":
    main()