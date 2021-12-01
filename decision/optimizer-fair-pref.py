"""
AI4G Decision Making Group
Fall 2021
Angel, Katherine, Matthew, Shashank, Zach, and Zhiming
"""

#For this package need to install C++14,cvxpy, and cvxopt
import cvxpy as cp
from decision.reward import *

num_iter = 10

#weight for fairness
w_fair = 10
#weight for preferences
w_preference = 1
#store allocation reward history
previous_allocation_rewards = []
allocation_history = []
#allocation time step
allocationNumber = 0

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

for iter in range(0,num_iter,1):
    past_tau = np.array(previous_allocation_rewards)
    # #demand
    # D = np.random.randint(1, 5, size=5).reshape(5, 1)
    # #number of users and items
    # U, I = D.shape
    # #supply
    # Gamma = np.random.randint(1,7, size=I)

    #Overleaf sample scenario
    # demand
    D = np.array([[1, 1, 0, 1, 1],[1,0,0,1,1]]).T
        #.reshape(5, 2)

    D2 = D.copy()
    D2 = np.where(D2 ==0, 100000,D)

    # number of users and items
    U, I = D.shape
    # supply
    Gamma = np.array([3,4])

    # score
    psi = np.array(compute_grade_threshold(D, Gamma))
    psi2 = np.array(([psi[0],psi[0],psi[0],psi[0],psi[0]],
                     [psi[1],psi[1],psi[1],psi[1],psi[1]])).T
    #psi = psi.reshape(2,1)
    # preference matrix randomly generated
    preference = np.random.rand(U, I).T

    if len(previous_allocation_rewards) == 0:
        # Variables
        allocation = cp.Variable((U, I), integer=True)
        delta = allocation / D2 - psi2
        tau = cp.sum(delta, axis=1)
        xi = tau
        alpha = cp.Variable()
    else:
        # Variables
        allocation = cp.Variable((U, I), integer=True)
        delta = allocation / D2 - psi2
        tau = cp.sum(delta, axis=1)
        xi = tau + cp.sum(gamma * past_tau, axis=0)
        alpha = cp.Variable()

    # Constraints
    constraints = []
    constraints.append(cp.sum(allocation, axis=0) <= Gamma)
    constraints.append(allocation <= D)
    constraints.append(alpha <= xi)
    constraints.append(allocation >=0)

    # Objective function
    objective = cp.Maximize(w_fair * alpha + w_preference * 1/(U*I)* cp.sum(allocation @ preference))

    # Problem
    prob = cp.Problem(objective, constraints)

    # Solve
    prob.solve()

    tauValue = tau.value

    idZero = np.where(D == 0)

    x,y = idZero
    for i in range(0,len(x)):
        tauValue[x[i]] += psi[y[i]]


    if len(previous_allocation_rewards) < 4:
        previous_allocation_rewards.append(tauValue)
    else:
        previous_allocation_rewards.pop(0)
        previous_allocation_rewards.append(tauValue)
    gamma,allocationNumber = gammaFunction(previous_allocation_rewards,allocationNumber)

    past_tau = np.array(previous_allocation_rewards)
    allocation_history.append(allocation.value)

    print("Allocation {}".format(allocationNumber))
    print("Demand")
    print(D)
    print("=================================================")
    print("Supply")
    print(Gamma)
    print("=================================================")
    print("Preference")
    print(preference)
    print("=================================================")
    print("Allocation")
    print(allocation.value)
    print("=================================================")
    print("Past Allocations Rewards")
    print(past_tau)
    print("Gamma")
    print(gamma)
    print("")
    print("Allocation History")
    print(allocation_history)

























