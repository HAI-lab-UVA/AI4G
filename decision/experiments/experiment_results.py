import numpy as np
import pandas as pd

import create_plots
import greedy_benchmark
import oracle_experiment as oe


def clean_learning_group_data(path: str) -> pd.DataFrame:
    """Cleans the Learning group data into the expected format for the other
    functions

    Args:
        path (str): Data path

    Returns:
        pd.DataFrame: Cleaned data
    """
    df = pd.read_csv(path)
    df = df.rename(columns={'ID': 'id'})
    df['id'] = df['id'].str.strip()

    # Rice (for whatever reason) has a space after before its .x where x 
    # indicates the day
    rice_idx = df.columns.str.startswith('Rice')
    no_space_rice_cols = [x.replace(' ', '') for x in df.columns[rice_idx]]
    rice_map = dict(zip(df.columns[rice_idx], no_space_rice_cols))
    df = df.rename(columns=rice_map)

    # The general format is ID, Oranges1, Apples1, ...
    # This needs to be changed from wide to long format and update the 
    # index scheme to match the assumed (id, day) format
    foods = list(set(match[0] for match in df.columns.str.findall(r'[^0-9]*')))
    foods.remove('id')
    foods.sort()
    df = pd.wide_to_long(df, foods, i='id', j='day')

    # Need to correct to zero indexing with the day from day 1 to day 0
    ids = df.index.levels[0].tolist()
    days = list(range(len(df.index.levels[1])))
    index = pd.MultiIndex.from_product([ids, days], names=('id', 'day'))
    df = df.set_index(index)

    # Finally the items are assumed to be in alphabetical order
    df = df.sort_index(axis=1)
    return df.sort_index()


if __name__ == '__main__':
    # Get the results for the "oracle" experiment first
    demand_oracle = pd.read_csv('results/demand.csv', index_col=[0, 1])
    mip_oracle = pd.read_csv('results/allocation_history.csv', index_col=[0, 1])
    greedy_oracle = pd.read_csv('results/greedy_allocation.csv', index_col=[0, 1])

    mip_df = create_plots.compute_frac_satisfied(mip_oracle, demand_oracle)
    greedy_df = create_plots.compute_frac_satisfied(greedy_oracle, demand_oracle)
    create_plots.create_satisfaction_plot(
        mip_df, greedy_df, 'results/demand-satisfaction-oracle.pdf', 'oracle'
    )

    create_plots.create_next_step_satisfaction_plot(
        mip_df, 'results/next-step-oracle.pdf'
    )

    # Next do the 14 day, no-feedback experiment
    supply_path = '../../learning/results/Predicted Supply.csv'
    demand_path = '../../learning/results/Predicted Demand.csv'
    supply_learned = clean_learning_group_data(supply_path)
    demand_learned = clean_learning_group_data(demand_path)

    user_names = oe.get_usernames('../../comms-1/data/user_responses.csv')
    food_names = supply_learned.columns
    preference_path = '../../comms-1/data/user_preferences.csv'
    P = oe.prepare_preferences(preference_path, user_names, food_names)
    P = oe.match_demand_and_preference(demand_learned, P)
    mip_learned, _ = oe.run_experiment(supply_learned, demand_learned, P)

    # allocation_path = '../Data/Part1/allocation_history.csv'
    # mip_learned = clean_allocation_data(allocation_path)
    mip_df = create_plots.compute_frac_satisfied(mip_learned, demand_oracle)
    greedy_learned = greedy_benchmark.greedy_experiment(supply_learned, demand_learned)
    greedy_df = create_plots.compute_frac_satisfied(greedy_learned, demand_oracle)

    # It's possible in this situation to allocate something even if the "true"
    # demand says the user didn't want anything -- we need to correct for this
    mip_df = mip_df.replace([np.inf, -np.inf], np.nan)
    greedy_df = greedy_df.replace([np.inf, -np.inf], np.nan)

    create_plots.create_satisfaction_plot(
        mip_df, greedy_df, 'results/demand-satisfaction-learned.pdf', 'learned'
    )
