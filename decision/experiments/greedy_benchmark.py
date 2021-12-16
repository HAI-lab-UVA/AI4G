import numpy as np
import pandas as pd

import oracle_experiment as oe


def greedy_allocation(S: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Implements a simple greedy allocation benchmark to compare against
    the MIP. It does this by satisfying user demand greedily without considering
    fairness, previous allocations, or user preference

    Args:
        S (np.ndarray): Supply matrix
        D (np.ndarray): Demand matrix

    Returns:
        np.ndarray: Allocation matrix
    """
    U, I = D.shape
    A = np.zeros(D.shape, dtype=int) # Assume format (users x items)
    S = S.sum(axis=0)

    for u in range(U):
        for i in range(I):
            if S[i] > 0: # If any supply is available
                A[u, i] = min(S[i], D[u, i])
                S[i] -= D[u, i]
            else:
                break # Nothing more to greedily allocate
    
    return A


def greedy_experiment(S: pd.DataFrame, D: pd.DataFrame) -> pd.DataFrame:
    """Runs greedy allocation experiment across all timesteps

    Args:
        S (pd.DataFrame): Supply data for all users and days
        D (pd.DataFrame): Demand data for all users and days

    Returns:
        pd.DataFrame: Allocation history for all users and days
    """
    users = np.array(D.index.levels[0]) # Assume (id, day) index
    items = D.columns
    df = pd.DataFrame()

    # At every step we need to permute the order of the users so we're not
    # always giving the first user everything they want
    rng = np.random.default_rng(17)

    for i in D.index.levels[1]:
        tmp_S = oe.get_data(S, i)
        tmp_D = oe.get_data(D, i)

        user_permutation = rng.permutation(np.arange(len(users)))
        tmp_users = users[user_permutation]

        tmp_A = greedy_allocation(tmp_S, tmp_D[user_permutation, :])
        df = oe.update_allocation_df(df, tmp_A, tmp_users, items, i)

    return df


if __name__ == '__main__':
    supply = pd.read_csv('results/supply.csv', index_col=[0, 1])
    demand = pd.read_csv('results/demand.csv', index_col=[0, 1])
    greedy_df = greedy_experiment(supply, demand)
    greedy_df.to_csv('results/greedy_allocation.csv')
