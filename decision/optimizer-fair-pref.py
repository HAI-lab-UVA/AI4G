"""
AI4G Decision Making Group
Fall 2021
Angel, Katherine, Matthew, Shashank, Zach, and Zhiming
"""

#For this package need to install C++14,cvxpy, and cvxopt
import cvxpy as cp
import mosek
import numpy as np
import pandas as pd

num_iter = 10

#weight for fairness
w_fair = 10
#weight for preferences
w_preference = 1
#store allocation reward history
previous_allocation_rewards = []
df_allocation_history = pd.DataFrame()
allocation_history = []
#allocation time step
allocationNumber = 0

allD = pd.read_csv("../learning/results/Predicted Demand.csv")
userID = allD.iloc[:, 0].to_numpy()
allD = allD.iloc[:, 1:allD.shape[1] - 1]

df_products = allD.iloc[:,0:19]

productsNames = list(df_products.columns)

productsNames = [i[:-1] for i in productsNames]

allGamma = pd.read_csv("../learning/results/Predicted Supply.csv")
allGamma = allGamma.iloc[:, 1:allGamma.shape[1]]

x, y = allD.shape


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


for i in range(0, y, 19):
    D = allD.iloc[:, i:i + 19].to_numpy()
    userID = userID[:]
    Gamma = allGamma.iloc[:, i:i + 19].to_numpy().sum(axis=0)

    past_tau = np.array(previous_allocation_rewards)
    D2 = D.copy()
    D2 = np.where(D2 ==0, 100000,D)

    # number of users and items
    U, I = D.shape

    # score
    psi2 = np.array(compute_grade_threshold(D, Gamma))

    # preference matrix randomly generated
    preference = np.random.rand(U, I).T

    # Define the variables for the problem
    allocation = cp.Variable((U, I), integer=True)
    delta = cp.Variable((U, I))
    tau = cp.Variable(U)
    xi = cp.Variable(U)
    alpha = cp.Variable()

    constraints = []
    constraints += [cp.sum(allocation, axis=0) <= Gamma]
    constraints += [allocation <= D]
    constraints += [delta == (allocation / D2) - psi2]
    constraints += [tau == cp.sum(delta, axis=1)]

    if len(previous_allocation_rewards) == 0:
        constraints += [xi == tau]
    else:
        constraints += [xi == tau + cp.sum(gamma * past_tau, axis=0)]
    
    constraints += [alpha <= xi]
    constraints += [allocation >= 0]

    # Objective function
    objective = cp.Maximize(w_fair * alpha + w_preference * 1 / (U * I) * cp.sum(allocation @ preference))

    # Problem
    prob = cp.Problem(objective, constraints)

    # Solve
    mosek_params = {mosek.dparam.optimizer_max_time: 10,
                    mosek.dparam.mio_tol_rel_gap: 0.05}
    prob.solve(solver=cp.MOSEK, mosek_params=mosek_params, verbose=True)

    tauValue = tau.value

    idZero = np.where(D == 0)

    x, y = idZero
    for i in range(0, len(x)):
        tauValue[x[i]] += psi2[0][y[i]]

    if len(previous_allocation_rewards) < 4:
        previous_allocation_rewards.append(tauValue)
    else:
        previous_allocation_rewards.pop(0)
        previous_allocation_rewards.append(tauValue)
    gamma, allocationNumber = gammaFunction(previous_allocation_rewards, allocationNumber)

    past_tau = np.array(previous_allocation_rewards)

    df_id = pd.DataFrame(userID,columns=["ID"])
    df_allocation = pd.DataFrame(allocation.value,columns = productsNames)

    df_id_allocation = pd.concat([df_id,df_allocation],axis=1)

    df_id_allocation["AllocationTime"] = allocationNumber

    df_allocation_history= df_allocation_history.append(df_id_allocation)

    #allocation_history.append(df_id_allocation.to_numpy())

    allocation_np = np.array(allocation_history)

    df_id_allocation.to_csv("./Data/allocation_day {}.csv".format(allocationNumber),sep=",",index=False)



    #allocation_history_csv = np.savetxt("./Data/allocation_history.csv", allocation_np[0], delimiter=",", fmt='%12.8f')

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


df_allocation_history.to_csv("./Data/allocation_history.csv".format(allocationNumber),sep=",",index=False)


























