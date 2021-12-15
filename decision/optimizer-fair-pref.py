"""
AI4G Decision Making Group
Fall 2021
Angel, Katherine, Matthew, Shashank, Zach, and Zhiming
"""
#Packages being used
#For this package need to install C++14,cvxpy, and cvxopt
import cvxpy as cp
#For this package need a mosek personal academic license which you can get with your UVA account
#https://www.mosek.com/products/academic-licenses/
import mosek
import numpy as np
import pandas as pd
import os
#allocation time step, start at 0
allocationNumber = 0
#weight for fairness
w_fair = 10
#weight for preferences
w_preference = 1


#read in allocation history
if os.path.exists("./Data/Part2/allocation_history.csv"):
    df_allocation_history = pd.read_csv("./Data/Part2/allocation_history.csv")
else:
    df_allocation_history = pd.DataFrame()

#read in reward history
if os.path.exists("./Data/Part2//previous_allocation_rewards.csv"):
    df_previous_allocation_rewards = pd.read_csv("./Data/Part2/previous_allocation_rewards.csv")
else:
    #store allocation reward history
    df_previous_allocation_rewards = pd.DataFrame()

#Read in data from other groups
allD = pd.read_csv("./PredictedDemandSupply/Predicted Demand_t8.csv")
userID = allD.iloc[:,0].to_numpy()
allD = allD.iloc[:, 1:allD.shape[1]]
#Learning Predicted Supply
allGamma = pd.read_csv("./PredictedDemandSupply/Predicted Supply_t8.csv")
allGamma = allGamma.iloc[:, 1:]


# #demand at t = 0
# #Learning Predicted Demand
# allD = pd.read_csv("../learning/results/Predicted Demand.csv")
# allD = allD.iloc[:,:21]
# userID = allD.iloc[:,0].to_numpy()
# allD = allD.iloc[:, 1:allD.shape[1] - 1]
# #Learning Predicted Supply
# allGamma = pd.read_csv("../learning/results/Predicted Supply.csv")
# allGamma = allGamma.iloc[:, 1:20]

#Communication 1 User Preferences
user_preferences = pd.read_csv("../comms-1/data/user_preferences.csv")
#User mapping
user_dict = {0:"NaN",
1:"Anonymous Narwhal",
2:"ak3gj",
3:"Gal Gadot",
4:"aj6eb",
5:"csk6snj",
6:"afv9x",
7:"pex7ps",
8:"zqz8ae",
9:"znp3ev",
10:"qwp4pk",
11:"nm2fs",
12:"blacklodge",
13:"ae3xd",
14:"Isobel Regina",
15:"kem5en",
16:"Broccoli"}

#create columns with user IDs
user_preferences["ID"] = user_preferences["Unnamed: 0"].map(user_dict)
#filter for IDs that are also in allD demand df
user_preferences = user_preferences.loc[user_preferences["ID"].isin(userID)]
#order needs to match that of demand df
user_preferences["ID"] = pd.Categorical(user_preferences["ID"],userID)
user_preferences2 = user_preferences.sort_values("ID")
#now that it is order we can remove the id columns
user_preferences2 = user_preferences2.iloc[:,1:-1]
#Replace -10000 values with 0 and then divide by 5 to get values between 0-1
user_preferences2 = user_preferences2.replace(-10000,0)/5
#Transpose the matrix to match our work
user_preferences2 = user_preferences2.transpose()
#Convert pandas df to np array
user_preferences_array = user_preferences2.to_numpy()

#Extract product names to later add as column names when exporting the data. Removing the 1 from the name
df_products = allD.iloc[:,0:19]
productsNames = list(df_products.columns)
productsNames = [i[:-1] for i in productsNames]

#get the number of users and items in the demand matrix
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
    if allocationNumber > 4 :
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

    previous_allocation_rewards = df_previous_allocation_rewards.to_numpy().tolist()
    gamma, allocationNumber = gammaFunction(previous_allocation_rewards, allocationNumber)

    past_tau = np.array(previous_allocation_rewards)
    #Dealing with division by 0
    D2 = D.copy()
    D2 = np.where(D2 ==0, 100000,D)

    # number of users and items
    U, I = D.shape

    # threshold score
    psi2 = np.array(compute_grade_threshold(D, Gamma))

    # user preference matrix
    # to randomly generate use np.random.rand(U, I).T
    preference = user_preferences_array

    # Variables
    allocation = cp.Variable((U, I), integer=True)
    delta = cp.Variable((U, I))
    tau = cp.Variable(U)
    xi = cp.Variable(U)
    alpha = cp.Variable()

    # Constraints
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
    objective = cp.Maximize(w_fair * alpha + w_preference *(1 / (U * I)) * cp.sum(allocation @ preference))

    # Problem
    prob = cp.Problem(objective, constraints)

    # Solve
    mosek_params = {mosek.dparam.optimizer_max_time: 10,
                    mosek.dparam.mio_tol_rel_gap: 0.05}
    prob.solve(solver=cp.MOSEK, mosek_params=mosek_params, verbose=False)

    #Dealing with division by 0
    #tau value
    tauValue = tau.value
    idZero = np.where(D == 0)

    x, y = idZero
    for i in range(0, len(x)):
        tauValue[x[i]] += psi2[0][y[i]]

    if len(previous_allocation_rewards) < 4:
        previous_allocation_rewards.append(tauValue)
        df_previous_allocation_rewards = pd.DataFrame(previous_allocation_rewards)
    else:
        previous_allocation_rewards.pop(0)
        previous_allocation_rewards.append(tauValue)
        df_previous_allocation_rewards = pd.DataFrame(previous_allocation_rewards)

    df_previous_allocation_rewards = pd.DataFrame(previous_allocation_rewards)

    #add ID columns and allocation time to allocation df df_id_allocation
    df_id = pd.DataFrame(userID,columns=["ID"])
    df_allocation = pd.DataFrame(allocation.value,columns = productsNames)
    df_id_allocation = pd.concat([df_id,df_allocation],axis=1)
    df_id_allocation["AllocationTime"] = allocationNumber+1
    #append allocation to allocation history
    df_allocation_history= df_allocation_history.append(df_id_allocation)

    #write to CSV files
    #allocation for time t
    df_id_allocation.to_csv("./Data/Part2/allocation_day{}_noFeedback.csv".format(allocationNumber+1),sep=",",index=False)
    #allocation history
    df_allocation_history.to_csv("./Data/Part2/allocation_history.csv".format(allocationNumber+1),sep=",",index=False)
    #reward history
    df_previous_allocation_rewards.to_csv("./Data/Part2/previous_allocation_rewards.csv".format(allocationNumber+1),sep=",",index=False)

    allocationNumber+=1
























