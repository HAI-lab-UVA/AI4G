import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_frac_satisfied(A: pd.DataFrame, D: pd.DataFrame) -> pd.DataFrame:
    """Computes the fraction of demand satisfied across all allocations

    Args:
        A (pd.DataFrame): Allocation DataFrame
        D (pd.DataFrame): Demand DataFrame

    Returns:
        pd.DataFrame: Demand satisfcation DataFrame
    """
    return A.divide(D)


def compute_next_step_satisfication(df: pd.DataFrame, col: str, sat_frac=1.) -> float:
    """Computes the fraction of time the system satisfied at least `sat_frac`
    demand after giving the user nothing the previous iteration

    Args:
        df (pd.DataFrame): DataFrame containing demand satisfication values
        col (str): Particular item (e.g., Apples)
        sat_frac (float, optional): Minimal demand satisfication value. Defaults to 1.

    Returns:
        float: Next-Step Satisfication value
    """

    indices = df.index[df[col] == 0.] # Have (id, day) multi-index

    # Possible that no indices meet this condition therefore this measure 
    # is irrelevant
    if len(indices) == 0:
        return np.nan

    count = 0
    last_day = max(df.index.levels[1])

    for idx in indices:
        # Have to check if it's the last day to avoid out-of-bound error
        if idx[1] == last_day:
            continue # By definition we cannot satisfy next day request
        else:
            frac = df[col][(idx[0], idx[1] + 1)]
            if frac >= sat_frac or np.isnan(frac):
                count += 1

    return count / len(indices)


def next_step_satisfication_by_user(df: pd.DataFrame, sat_frac: float) -> pd.DataFrame:
    """Computes next-step satisfcation on a per-user basis

    Args:
        df (pd.DataFrame): DataFrame containing demand satisfication values
        sat_frac (float): Desired minimal percentage of next-step satisfcation

    Returns:
        pd.DataFrame: Next-Step satisfaction DataFrame on per-user basis
    """

    next_step = {}
    users = df.index.levels[0] # Have multi-index with (id, day) format
    cols = df.columns

    for user in users:
        tmp_df = df.query('id == @user')
        vals = np.zeros(len(cols))
        for (i, col) in enumerate(cols):
            vals[i] = compute_next_step_satisfication(tmp_df, col, sat_frac)
        
        next_step[user] = np.nanmean(vals)

    return pd.DataFrame.from_dict(next_step, orient='index', columns=['fraction'])


def create_satisfaction_plot(mip_df: pd.DataFrame, greedy_df: pd.DataFrame, path: str,
                             case: str):
    """Creates bar plot showing demand satisfaction comparison between the
    MIP and greedy allocation strategy

    Args:
        mip_df (pd.DataFrame): MIP demand satisfaction DataFrame
        greedy_df (pd.DataFrame): Greedy demand satisfaction DataFrame
        path (str): Location to save plot
        case (str): Oracle or Learned distribution case
    """
    mip_median = mip_df.groupby('id').mean().median(axis=1) # Id on axis=1
    greedy_median = greedy_df.groupby('id').mean().median(axis=1)

    index = mip_median.index
    y = np.arange(len(index))
    height = 0.35

    _, ax = plt.subplots(figsize=(12, 9))
    bar1 = ax.barh(y + height/2, mip_median, height, color='#beaed4', label='MIP')
    bar2 = ax.barh(y - height/2, greedy_median, height, color='#fdc086', label='Greedy')
    ax.bar_label(bar1, fmt='%.2f', padding=3, fontsize=14)
    ax.bar_label(bar2, fmt='%.2f', padding=3, fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_yticks(y, index.tolist())

    if case == "oracle":
        title = 'Median Demand Satisfaction by Customer\n (Oracle Knowledge)'
    else:
        title = 'Median Demand Satisfaction by Customer\n (Learned Distributions)'

    ax.set_title(title, fontsize=24)
    ax.legend()
    plt.savefig(path, dpi=300, bbox_inches='tight')


def create_next_step_satisfaction_plot(mip_df: pd.DataFrame, 
                                       path: str):
    """Creates plot showing either full or partial demand satisfaction for the 
    next step after getting nothing

    Args:
        mip_df (pd.DataFrame): MIP demand satisfaction DataFrame
        sat_frac (float): Minimal demand satisfication value
        path (str): Location to save plot
    """
    mip_full = next_step_satisfication_by_user(mip_df, 1.)
    mip_partial = next_step_satisfication_by_user(mip_df, 0.01)
    index = mip_full.index
    y = np.arange(len(index))
    height = 0.35

    _, ax = plt.subplots(figsize=(12, 9))
    bar1 = ax.barh(y + height/2, mip_full.fraction, height, color='#beaed4', label='Full')
    bar2 = ax.barh(y - height/2, mip_partial.fraction, height, color='#fdc086', label='Partial')
    ax.bar_label(bar1, fmt='%.2f', padding=3, fontsize=14)
    ax.bar_label(bar2, fmt='%.2f', padding=3, fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_yticks(y, index.tolist())
    ax.set_title('Median Next-Step Customer Satisfication', fontsize=24)
    ax.legend()
    plt.savefig(path, dpi=300, bbox_inches='tight')
