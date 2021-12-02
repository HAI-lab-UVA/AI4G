import argparse
from numpy import genfromtxt
import pandas as pd

from feedback_module import Feedback


def main():

    parser = argparse.ArgumentParser(description='Feedback module')
    parser.add_argument('decision', help='Decision Matrix data in csv form')
    subparsers = parser.add_subparsers(help='')
    sp = subparsers.add_parser('1', help='Single Feedback csv containing both Allocation and Item feedback')
    sp.set_defaults(cmd = '1')
    sp.add_argument('-f', '--feedback', help='Feedback survey data in csv form')
    sp = subparsers.add_parser('2', help='Separate Allocation and Item feedback csv files')
    sp.set_defaults(cmd = '2')
    sp.add_argument('-a', '--alloc', help='Allocation Feedback data in csv form')
    sp.add_argument('-i', '--item', help='Item Feedback data in csv form')

    parser.add_argument('usersim', help='User similarity data in csv form')
    parser.add_argument('itemsim', help='Item similarity data in csv form')
    parser.add_argument('-o', '--outfile', default="predicted_feedback",  help='output csv filename')

    args = parser.parse_args()

    # if not args.decision:
    #     print ("Decision data was not specified")
    #     exit()
    # if (not args.feedback) and not (args.alloc and args.item):
    #     print ("Feedback data was not specified")
    #     exit()
    # if (not args.usersim):
    #     print ("User similarity data was not specified")
    # if (not args.itemsim):
    #     print ("Item similarity data was not specified")

    decision_df = pd.read_csv(args.decision)

    alloc_feedback_df = None
    item_feedback_df = None
    if args.feedback:
        pass
    elif args.alloc and args.item:
        alloc_feedback_df = pd.read_csv(args.alloc)
        item_feedback_df = pd.read_csv(args.item)

    usersim_df = genfromtxt(args.usersim, delimiter=',')
    itemsim_df = genfromtxt(args.itemsim, delimiter=',')

    f = Feedback()

    feedback_prediction_df = f.predict_feedback(decision_df, item_feedback_df, alloc_feedback_df, usersim_df)

    feedback_prediction_df.to_csv("{}.csv".format(args.outfile), index=False)


if __name__ == "__main__":
    main()