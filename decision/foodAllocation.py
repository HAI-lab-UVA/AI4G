"""
AI4G Decision Making Group
Food Allocation script that accounts for fairness and preferences
Fall 2021
Angel, Katherine, Matthew, Shashank, Zach, and Zhiming
"""

#For this package need to install C++14,cvxpy, and cvxopt
import cvxpy as cp
import numpy as np

#weight for fairness
w_fair = 10
#weight for preferences
w_preference = 1
#store allocation and reward history
rewards_history = []
allocation_history = []
#allocation time step
allocationNumber = 0

def allocateFood(D,Gamma,preferences,rewards_history):
    '''
    Function that generates the food allocation
    :param D: Estimated demand matrix
    :param Gamma: Estimated supply matrix
    :param preferences: Preference matrix
    :return:
    '''
    past_tau = np.array(rewards_history)

    # To deal with the division by 0 error, we create a copy of the demand
    # and replace 0 with a large number
    # create copy of demand D2
    D2 = D.copy()
    # replace 0 values with large number
    D2 = np.where(D2 == 0, 100000, D)

    # number of users and items
    U, I = D.shape

    # score
    psi = np.array(compute_grade_threshold(D, Gamma))

    if len(rewards_history) == 0:
        # Variables
        allocation = cp.Variable((U, I), integer=True)
        delta = allocation / D2 - psi
        tau = cp.sum(delta, axis=1)
        xi = tau
        alpha = cp.Variable()
    else:
        # Variables
        allocation = cp.Variable((U, I), integer=True)
        delta = allocation / D2 - psi
        tau = cp.sum(delta, axis=1)
        xi = tau + cp.sum(gamma * past_tau, axis=0)
        alpha = cp.Variable()

    #Optimization Problem
    # Constraints
    constraints = []
    constraints.append(cp.sum(allocation, axis=0) <= Gamma)
    constraints.append(allocation <= D)
    constraints.append(alpha <= xi)
    constraints.append(allocation >=0)

    # Objective function
    objective = cp.Maximize(w_fair * alpha + w_preference * 1/(U*I)* cp.sum(allocation @ preferences))

    # Problem
    prob = cp.Problem(objective, constraints)

    # Solve
    prob.solve()

    #users whose demand for item x was 0 should get a tau value of 0
    #adding back the psi value to those users
    tauValue = tau.value
    #Identify where 0 exists in the D matrix
    idZero = np.where(D == 0)

    #index where 0 exists in the D matrix
    x,y = idZero
    #add back the psi value to tau for those users
    for i in range(0,len(x)):
        tauValue[x[i]] += psi[0][y[i]]

    #append the reward to the reward history matrix
    if len(rewards_history) < 4:
        rewards_history.append(tauValue)
    else:
        #we are only storing 4 iterations
        rewards_history.pop(0)
        rewards_history.append(tauValue)
    gamma,allocationNumber = gammaFunction(rewards_history,0)

    past_tau = np.array(rewards_history)
    allocation_history.append(allocation.value)

    return allocation.value


def compute_grade_threshold(D,Gamma):
    '''
    The threshold values are computed by item for a given instance of the
    D and Gamma matrices
    :param D: Estimated demand matrix
    :param Gamma: Estimated supply matrix
    :return: threshold (psi)
    '''
    total_demand = D.sum(axis=0)
    threshold = Gamma / total_demand
    threshold[threshold >= 1] = 1
    # number of users and items
    U, I = D.shape

    #create a new array of U by I to work with cvxpy
    threshold2 = np.tile(threshold,(U,1))

    return threshold2

def gammaFunction(previous_allocation_rewards, allocationNumber):
    """
    Function that generates the gamma discounting vector
    Once the previous allocation history size is > 4
    Then the gamma vector will be [1/4 2/4 3/4 4/4]
    :param previous_allocation_rewards: array storing the past allocation rewards
    :param allocationNumber: allocation time step
    :return: gamma vector and the allocation time step
    """
    allocationNumber += 1
    if allocationNumber > 3 :
        gamma = []
        for i in range(4):
            gamma.append((i + 1) / 4)
        gammaArray = np.array(gamma)
        return gammaArray.reshape(len(gammaArray), 1), allocationNumber
    else:
        gamma = []
        for i in range(len(previous_allocation_rewards)):
            gamma.append((i + 1) / allocationNumber)
        gammaArray = np.array(gamma)
        return gammaArray.reshape(len(gammaArray),1), allocationNumber

def simulate_data(U, I, low, high):
    '''
    Simulated data for testing food allocation algorithm
    :param U: number of users (20)
    :param I: number of food items (20)
    :param low: lower bound (0)
    :param high: upper bound (10 for demand, 7 for supply)
    :return: a 20x20 array
    '''
    np.random.seed(1)
    return np.random.randint(low=low, high=high, size=(U, I), dtype=int)

if __name__ == '__main__':
    #Considering that we have 20 users and 20 items

    #Simulated Estimated Demand
    D = simulate_data(20, 20, 0, 5)
    #Simulated Estimated Supply
    Gamma = simulate_data(1, 20, 0, 30)[0]


    # number of users and items
    U, I = D.shape

    #Simulated Preferences
    preferences = np.random.rand(U, I).T

    allocation = allocateFood(D,Gamma,preferences,rewards_history)

    print("Allocation {}".format(allocationNumber))
    print("Demand")
    print(D)
    print("=================================================")
    print("Supply")
    print(Gamma)
    print("=================================================")
    print("Preference")
    print(preferences)
    print("=================================================")
    print("Allocation")
    print(allocation)
    print("=================================================")
    print("Past Allocations Rewards")
    print(rewards_history)
    print("")
    print("Allocation History")
    print(allocation_history)


