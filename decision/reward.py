"""
AI4G Decision Making Group
Fall 2021
Angel, Katherine, Matthew, Shashank, Zach, and Zhiming
"""

import numpy as np


def compute_grade_threshold(D,Gamma):
    # The threshold values are computed by item for a given instance of the
    # H and D matrices

    total_demand = D.sum(axis=0)
    threshold = Gamma / total_demand
    threshold[threshold >= 1] = 1
    return threshold


def compute_allocation_score(B, H, threshold):
    diff = B - threshold

    # It's possible that no items were requested and none were given hence
    # I want to ignore the divide by zero error; this is handled later
    with np.errstate(divide='ignore', invalid='ignore'):
        frac_satisfied = B / H

    satisfied_diff = threshold - frac_satisfied

    score_map = np.where(diff >= 0, 1, -1)
    idx = np.where(score_map == -1)

    scores = np.ones(B.shape)
    scores[idx] = -1 * satisfied_diff[idx]

    test = np.nansum(scores, axis=1).mean()
    # Propose scoring allocation by user and then taking the average
    return np.nansum(scores, axis=1).mean()


def compute_preference_score(B, P):
    return (B * P).sum(axis=1).mean()


def compute_reward(w, scores):
    return np.dot(w, scores)


if __name__ == '__main__':
    D = np.array([[1, 10]])
    H = np.array([[2, 2, 2, 2, 2], [1, 2, 0, 1, 0]]).T
    B = np.array([[1, 0, 0, 0, 0], [1, 2, 0, 1, 0]]).T

    print(D)
    print(H)
    print(B)

    threshold = compute_grade_threshold(H, D)
    print(threshold)
    print(compute_allocation_score(B, H, threshold))
