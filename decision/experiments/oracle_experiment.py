import cvxpy as cp
import mosek
import numpy as np
import os
import pandas as pd
import re


def prepare_data(path: str):
    df = pd.read_csv(path)
    df = df.drop(columns=['Timestamp'])

    name_str = ('Please enter an identifier (ex. computing id). '
                'Please use the same identifier as you used on '
                'the previous survey.')

    df = df.rename(columns={name_str: 'id'})
    df['id'] = df['id'].str.strip()

    # The surveys were done by asking all 14 days at once, so for example
    # eggs are represented as Eggs, ..., Eggs.13
    food_str = 'Please indicate the number of: '
    df = df.rename(columns=lambda x: re.sub(food_str, '', x))

    # Rice (for whatever reason) has a space after before its .x where x 
    # indicates the day
    rice_idx = df.columns.str.startswith('Rice')
    no_space_rice_cols = [x.replace(' ', '') for x in df.columns[rice_idx]]
    rice_map = dict(zip(df.columns[rice_idx], no_space_rice_cols))
    df = df.rename(columns=rice_map)

    # To use the wide-to-long function in Pandas we need to add a "0" to the
    # first set of food items to make it clear that's the delimiter
    next_day_idx = df.columns.get_loc('Oranges.1')
    updated_cols = [x + '.0' for x in df.columns[1:next_day_idx]]
    col_map = dict(zip(df.columns[1:next_day_idx], updated_cols))
    df = df.rename(columns=col_map)

    foods = list(set(match[0] for match in df.columns.str.findall(r'[^\.]*')))
    foods.remove('id')
    foods.sort()
    df = pd.wide_to_long(df, foods, i='id', j='day', sep='.')

    # By assumption a non-response corresponds to a zero
    df = df.fillna(0)
    df = df.astype(int) # By assumption everything is an integer
    return df


def get_usernames(path: str):
    df = pd.read_csv(path)

    name_str = 'Please enter an identifier (ex. computing id)'
    names = df[name_str].values
    
    # For whatever reason the first user does not have a username
    names[0] = 'Unnamed'
    return names


def prepare_preferences(path: str, user_names, food_names):
    df = pd.read_csv(path)
    
    # The first column corresponding to the users currently has no name
    # and only has integer values (e.g., 0, 1, ...)
    df = df.rename(columns={'Unnamed: 0': 'id'})
    df['id'] = user_names
    df = df.set_index('id')
    
    # We also need to ensure food naming consistency across the supply, demand
    # and preference matrices so we have a common output style
    df = df.sort_index(axis=1)
    col_map = dict(zip(df.columns, food_names))
    df = df.rename(columns=col_map)
    return df


def match_demand_and_preference(D: pd.DataFrame, P: pd.DataFrame):
    # Everyone filled out a preference matrix, but only some of them are 
    # demanding food so we need to match the preference matrix accordingly
    demand_ids = D.index.levels[0]
    preference_ids = P.index
    common_names = np.intersect1d(demand_ids, preference_ids)
    P = P.query('id.isin(@common_names)').sort_index()
    return P


def compute_ψ(S: np.ndarray, D: np.ndarray):
    total_S = S.sum(axis=0)
    total_D = D.sum(axis=0)
    ψ = total_S / total_D
    ψ[ψ >= 1.] = 1.
    return ψ


def get_data(df: pd.DataFrame, day: int):
    df = (df.query('day == @day')
            .reset_index(level='day', drop=True)
            .sort_index()
            .to_numpy())

    return df


def make_allocation(S, D, P, ψ, w=(10, 1), γ=None, τ_prev=None, optimality_gap=0.05,
                    max_solve_time=10):
    """Makes optimal allocations using MIP formulation

    Args:
        S (np.ndarray): Supply
        D (np.ndarray): Demand
        P (np.ndarray): Preferences
        ψ (np.ndarray): Demand threshold values (e.g., can only satisfy 70% demand)
        w (tuple, optional): Weights of fairness and preference satisfcation
        γ (np.ndarray): Previous allocation penalty factor (e.g., 1/2, 1/4, etc.)
        τ_prev (np.ndarray, optional): Previous allocation rewards
        optimality_gap (float, optional): Defaults to 5%
        max_solve_time (int, optional): Defaults to 10 seconds

    Returns:
        A, τ, and α
    """

    # If a user has a strong negative preference (-10000) for an item they 
    # should not receive anything and their demand should also be 0
    idx = np.where(P == -10000)
    D[idx] = 0.
    P[idx] = 0.
    P /= 5. # Scales all values between [0-1]

    U, I = D.shape
    A = cp.Variable((U, I), integer=True)
    δ = cp.Variable((U, I))
    τ = cp.Variable(U)
    ξ = cp.Variable(U)
    α = cp.Variable()

    # We only care about the total supply for each item; how we get it is 
    # irrelevant from our perspective
    S = S.sum(axis=0)

    constraints = []
    constraints += [A >= 0]
    constraints += [cp.sum(A, axis=0) <= S] # Can't exceed available supply
    constraints += [A <= D] # Don't give more than they requested

    # We reward allocation based on available supply while accounting for the
    # possibility the D[u, i] was 0 and hence should receive no reward or 
    # penalty because nothing will be allocated
    for i in range(I): 
        for u in range(U):
            if D[u, i] == 0:
                constraints += [δ[u, i] == 0]
            else:
                constraints += [δ[u, i] == (A[u, i] / D[u, i]) - ψ[i]]
    
    # This gets the max-min user vector for our fairness criteria
    constraints += [τ == cp.sum(δ, axis=1)]

    # We account for previous allocations so we are not systematically giving
    # a user less than they request
    if τ_prev is None:
        constraints += [ξ == τ]
    else:
        constraints += [ξ == τ + cp.sum(γ[:, None] * τ_prev, axis=0)]
    
    constraints += [α <= ξ]

    # We consider a multi-objective problem of maximizing fairness & preference
    fairness = w[0] * α
    preference = w[1] * (1 / (U * I)) * cp.sum(cp.multiply(A, P))
    objective = cp.Maximize(fairness + preference)
    prob = cp.Problem(objective, constraints)

    # To help the solver we also define an acceptable optimality gap and
    # solve time
    mosek_params = {mosek.dparam.optimizer_max_time: max_solve_time,
                    mosek.dparam.mio_tol_rel_gap: optimality_gap}
    prob.solve(solver=cp.MOSEK, mosek_params=mosek_params, verbose=False)
    return A.value, τ.value, α.value


def update_allocation_df(df: pd.DataFrame, A: np.ndarray, users: list, 
                         items: list, day: int):
    """Updates the allocation history DataFrame
    """

    index = zip(users, [day for _ in range(len(users))])
    index = pd.MultiIndex.from_tuples(index, names=['id', 'day'])
    new_df = pd.DataFrame(data=A, columns=items, index=index)
    df = df.append(new_df)
    return df


def compute_γ(day: int):
    if day >= 4:
        return np.array([0.25, 0.50, 0.75, 1.])
    elif day == 1:
        return np.array([1.])
    elif day == 2:
        return np.array([0.50, 1.])
    else:
        return np.array([0.33, 0.67, 1.])


def make_obj_vals_df(obj_vals):
    index = pd.Index(range(14), name='day')
    df = pd.DataFrame(obj_vals, columns=['objective_value'], index=index)
    return df


def run_experiment(supply: pd.DataFrame, demand: pd.DataFrame, P: pd.DataFrame):
    """Runs oracle experiment across 14 days assuming perfect knowledge of
    supply and demand

    Args:
        supply (pd.DataFrame): True supply data
        demand (pd.DataFrame): True demand data
        P (pd.DataFrame): User preferences

    Returns:
        tuple: Allocation History and Objective Values
    """
    users = P.index
    items = demand.columns
    P = P.to_numpy()
    obj_vals = []
    allocation_history = pd.DataFrame()
    τ_prev = np.empty((0, len(users)))

    for i in supply.index.levels[1]:
        S = get_data(supply, i)
        D = get_data(demand, i)
        ψ = compute_ψ(S, D)

        if i == 0:
            A, τ, α = make_allocation(S, D, P, ψ)
            τ = τ.reshape(1, -1)
        else:
            γ = compute_γ(i)
            if τ_prev.shape[0] > 4:
                τ_prev = np.delete(τ_prev, 0, 0)
            
            A, τ, α = make_allocation(S, D, P, ψ, γ=γ, τ_prev=τ_prev)
            τ = τ.reshape(1, -1)

        allocation_history = update_allocation_df(
            allocation_history, A, users, items, i
        )

        obj_vals.append(α)
        τ_prev = np.append(τ_prev, τ, axis=0)
    
    obj_vals = make_obj_vals_df(obj_vals)
    return allocation_history, obj_vals


if __name__ == '__main__':
    supply = prepare_data('../../learning/supply survey.csv')
    demand = prepare_data('../../learning/demand survey.csv')

    user_names = get_usernames('../../comms-1/data/user_responses.csv')
    food_names = supply.columns
    preference_path = '../../comms-1/data/user_preferences.csv'
    P = prepare_preferences(preference_path, user_names, food_names)
    P = match_demand_and_preference(demand, P)

    allocation_hist, obj_vals = run_experiment(supply, demand, P)

    if not os.path.exists('results'):
        os.mkdir('results')
    
    allocation_hist.to_csv('results/allocation_history.csv')
    supply.to_csv('results/supply.csv')
    demand.to_csv('results/demand.csv')
    obj_vals.to_csv('results/obj_vals.csv')
