
from __future__ import (absolute_import, division, generator_stop, print_function,
                        unicode_literals)
import csv, random, os
import numpy as np
from pandas.core.algorithms import value_counts
from six.moves import range
from six import iteritems
from surprise import AlgoBase, Reader, Dataset
from surprise import PredictionImpossible
from surprise.utils import get_rng

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, paired_euclidean_distances
from sklearn.preprocessing import normalize


class SVDpp_neighborhood(AlgoBase):

    def __init__(self, data_path, train_split = ("random", .9), min_ratings = 1, n_factors=20, n_epochs=20, init_mean=0, init_std_dev=.1,
                 lr_all=.007, reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None,
                 lr_qi=None, lr_xj=None, lr_yj=None, lr_zv=None, reg_bu=None, reg_bi=None, reg_pu=None,
                 reg_qi=None, reg_xj=None, reg_yj=None, reg_zv=None, random_state=None, verbose=False):

        self.min_ratings = min_ratings
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.lr_xj = lr_xj if lr_xj is not None else lr_all
        self.lr_yj = lr_yj if lr_yj is not None else lr_all
        self.lr_zv = lr_zv if lr_zv is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.reg_xj = reg_xj if reg_xj is not None else reg_all
        self.reg_yj = reg_yj if reg_yj is not None else reg_all
        self.reg_zv = reg_zv if reg_zv is not None else reg_all
        self.random_state = random_state
        self.verbose = verbose

        self.bu = self.bi = self.pu = self.qi = self.xj = self.yj = self.zv = self.pre_sum_i = self.pre_sum_u = None

        self.user_sim = self.item_sim = None
        self.been_fit = False

        parsed_responses, user_mappings = dataset(data_path)

        self.user_mapping = user_mappings.rename(columns = {"Please enter an identifier (ex. computing id)" : ""})
        self.data = parsed_responses

        pref_columns = ["user_id", "pref_orange", "pref_apple", "pref_watermelon", "pref_banana", "pref_eggplant", "pref_tomatoes", "pref_potatoes",
                        "pref_bread", "pref_oats", "pref_rice", "pref_fish", "pref_egg", "pref_chicken", "pref_olive_oil", "pref_soybean_paste", "pref_beef",
                        "pref_milk", "pref_yogurt", "pref_cheese_ball"]

        self.pref = self.data[pref_columns]


        # Creating our own train/test split, since the surprise one is bad.

        if train_split[0] == "random":
            ratings = pd.melt(self.pref, id_vars = pref_columns[0], value_vars = pref_columns[1:], var_name = 'food_item',
                              value_name = 'rating_given').dropna().sample(frac = 1, random_state = self.random_state)

            split_point = int(len(ratings)*train_split[1])
            ratings_train = ratings[["user_id","food_item","rating_given"]][:split_point]
            ratings_test = ratings[["user_id","food_item","rating_given"]][split_point:]

        elif train_split[0] == 'rows':
            ratings = self.pref.sample(frac = 1, random_state = self.random_state)

            ratings_train = pd.melt(ratings[:-train_split[1]], id_vars = pref_columns[0], value_vars = pref_columns[1:],
                                    var_name = 'food_item', value_name = 'rating_given').dropna()
            ratings_test = pd.melt(ratings[-train_split[1]:], id_vars = pref_columns[0], value_vars = pref_columns[1:],
                                   var_name = 'food_item', value_name = 'rating_given').dropna()

        else:
            ratings = self.pref[self.pref.columns[1:]].sample(frac = 1, axis = 'columns', random_state = self.random_state)
            ratings.insert(loc = train_split[1], column = "user_id", value = self.pref[self.pref.columns[0]]) # I know this is really janky, but it works... don't @ me lol

            ratings_train = pd.melt(ratings[ratings.columns[train_split[1]:]], id_vars = pref_columns[0], value_vars = ratings.columns[train_split[1]+1:],var_name = 'food_item', value_name = 'rating_given').dropna()
            ratings_test = pd.melt(ratings[ratings.columns[:train_split[1]+1]], id_vars = pref_columns[0], value_vars = ratings.columns[:train_split[1]:], var_name = 'food_item', value_name = 'rating_given').dropna()



        reader = Reader(rating_scale=(0, 5))
        ds = Dataset.load_from_df(ratings_train[["user_id","food_item","rating_given"]], reader)
        self.trainset = ds.build_full_trainset()
        test_set = Dataset.load_from_df(ratings_test[["user_id","food_item","rating_given"]], reader)
        self.testset = test_set.build_full_trainset().build_testset()

        AlgoBase.__init__(self)

    def fit(self, n_epochs = None):
        n_epochs = self.n_epochs if n_epochs == None else n_epochs
        self.been_fit = True
        AlgoBase.fit(self, self.trainset)
        self.sgd(self.trainset, n_epochs)
        self.sim_mat(is_user = True)
        self.sim_mat(is_user = False)

        return self

    def sgd(self, trainset, n_epochs):

        bi = self.bi if not self.bi is None else np.zeros(trainset.n_items) # item biases
        bu = self.bu if not self.bu is None else np.zeros(trainset.n_users) # user biases

        rng = get_rng(self.random_state)

        qi = self.qi if not self.qi is None else rng.normal(self.init_mean, self.init_std_dev, (trainset.n_items, self.n_factors))
        xj = self.xj if not self.xj is None else rng.normal(self.init_mean, self.init_std_dev, (trainset.n_items, self.n_factors))
        yj = self.yj if not self.yj is None else rng.normal(self.init_mean, self.init_std_dev, (trainset.n_items, self.n_factors))
        pre_sum_u = self.pre_sum_u if not self.pre_sum_u is None else np.zeros((trainset.n_users, self.n_factors))

        pu = self.pu if not self.pu is None else rng.normal(self.init_mean, self.init_std_dev, (trainset.n_users, self.n_factors))
        zv = self.zv if not self.zv is None else rng.normal(self.init_mean, self.init_std_dev, (trainset.n_users, self.n_factors))
        pre_sum_i = self.pre_sum_i if not self.pre_sum_i is None else np.zeros((trainset.n_items, self.n_factors))



        for current_epoch in range(n_epochs):
            if self.verbose:
                print(" processing epoch {}".format(current_epoch))
            for u, R_u in iteritems(trainset.ur):

                sqrt_R_u = np.sqrt(len(R_u))

                pre_sum_u[u] = (sum((r - self.trainset.global_mean + bu[u] + bi[j])*xj[j] + yj[j] for j, r in R_u) / sqrt_R_u)

                total_e =  0

                for i, r in R_u:
                    r_hat = self.trainset.global_mean + bu[u] + bi[i] + np.dot(qi[i], pre_sum_u[u])
                    err = (r - r_hat)
                    total_e += err*qi[i]

                    bu[u] += self.lr_bu * (err - self.reg_bu * bu[u])
                    bi[i] += self.lr_bi * (err - self.reg_bi * bi[i])

                    qi[i] += self.lr_qi * (err * pre_sum_u[u] - self.reg_qi * qi[i])

                for j, r in R_u:
                    buj = self.trainset.global_mean + bu[u] + bi[j]
                    xj[j] += self.lr_yj * ((r - buj) * total_e / sqrt_R_u - self.reg_yj * yj[j])
                    yj[j] += self.lr_yj * (total_e / sqrt_R_u - self.reg_yj * yj[j])

            for i, R_i in iteritems(trainset.ir):

                sqrt_R_i = np.sqrt(len(R_i))

                pre_sum_i[i] = (sum((r - self.trainset.global_mean + bu[v] + bi[i])*zv[v] for v, r in R_i) / sqrt_R_i)

                total_e =  0

                for u, r in R_i:
                    r_hat = self.trainset.global_mean + bu[u] + bi[i] + np.dot(pu[u], pre_sum_i[i])
                    err = (r - r_hat)
                    total_e += err*pu[u]

                    bu[u] += self.lr_bu * (err - self.reg_bu * bu[u])
                    bi[i] += self.lr_bi * (err - self.reg_bi * bi[i])

                    pu[u] += self.lr_pu * (err * pre_sum_i[i] - self.reg_pu * pu[u])

                for v, r in R_i:
                    bvi = self.trainset.global_mean + bu[v] + bi[i]
                    zv[v] += self.lr_zv * ((r - bvi) * total_e / sqrt_R_i - self.reg_zv * zv[v])

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi
        self.xj = xj
        self.yj = yj
        self.zv = zv
        self.pre_sum_u = pre_sum_u
        self.pre_sum_i = pre_sum_i

    def estimate(self, u, i):
        est = self.trainset.global_mean

        # breakpoint()
        if len(self.trainset.ur[u]) > self.min_ratings:
            est += self.bu[u]
        else:
            userTemp = 0

            for user in self.trainset.ur:
                itemTemp = 0
                for item, rating in self.trainset.ur[user]:
                    # breakpoint()
                    sim = self.user_sim[user][int(u[-1])] if type(u) == str else self.user_sim[user][u]
                    itemTemp += (rating - self.bu[user] - self.bi[item] - self.trainset.global_mean) * sim

                itemTemp /= np.sqrt(len(self.trainset.ur[user])) if len((self.trainset.ur[user])) > 0 else 1
                # print(u,i,user,itemTemp)
                userTemp += itemTemp

            try:
                userTemp /= len(self.trainset.ur)
            except:
                breakpoint()
            est += userTemp

        if len(self.trainset.ir[i]) > self.min_ratings:
            est += self.bi[i]
        else:
            itemTemp = 0
            for item in self.trainset.ir:
                userTemp = 0

                for user, rating in self.trainset.ir[item]:
                    sim = self.item_sim[item][self.mapping_item[i[10:]]] if type(i) == str else self.item_sim[item][i]
                    # userTemp += (rating - self.bu[user] - self.bi[item] - self.trainset.global_mean) * self.item_sim[item][pd.Index(self.item_mapping).get_loc(i[10:])]
                    userTemp += (rating - self.bu[user] - self.bi[item] - self.trainset.global_mean) * sim

                userTemp /= np.sqrt(len(self.trainset.ir[item])) if len((self.trainset.ir[item])) > 0 else 1
                itemTemp += userTemp
            itemTemp /= len(self.trainset.ir)
            est += itemTemp


        if len(self.trainset.ur[u]) > self.min_ratings and len(self.trainset.ir[i]) > self.min_ratings:
            sqrt_R_u = np.sqrt(len(self.trainset.ur[u]))  # nb of items rated by u
            try:
                item_based = np.dot(self.qi[i], self.pre_sum_u[u])
            except:
                breakpoint()
            user_based = np.dot(self.pu[u], self.pre_sum_i[i])
            est += item_based + user_based


        return est

    def sim_mat(self, as_df = False, write_out = False, is_user = True):
        if is_user:
            if self.user_sim is None:
                df = self.data
                user_sim = np.zeros((df.shape[0],df.shape[0]))

                A = df.to_numpy()
                L2_norm = np.sqrt(np.nansum(A**2, axis=0).astype('float'))
                L2_normT = np.array([1/val if not val == 0 else 0 for val in L2_norm ])
                A_L2 = (A.transpose() * L2_normT[:,np.newaxis]).transpose()

                df = pd.DataFrame(A_L2)

                for user1 in df.iterrows():
                    for user2 in df.iterrows():
                        try:
                            user_sim[user1[0]][user2[0]] = nan_cosine_similarity(np.array(user1[1][1:]), np.array(user2[1][1:]))
                        except:
                            breakpoint()

                self.user_sim = user_sim

            if as_df:
                df_out = pd.DataFrame(data = self.user_sim, index = self.user_mapping, columns = self.user_mapping)
                if write_out:
                    df_out.to_csv("data\\user_sim.csv")
                return df_out

            else:
                if write_out:
                    np.savetxt("data\\user_sim_np.csv", self.user_sim, delimiter=",")

            return self.user_sim

        else:
            if self.item_sim is None:
                path='data/item_list.csv'
                df=pd.read_csv(path).set_index('Item_name')
                self.item_mapping = pd.Series(df.index)
                self.mapping_item = {index:val for val,index in enumerate(self.item_mapping.to_numpy())} #to make estimate faster
                self.item_mapping.name = ''

                df.drop(['Notes/comments'],axis=1,inplace=True)
                df['calories'] = df['calories'].astype('float')
                for c in df.columns[5:10]:
                    df[c] = df[c].astype('float')

                for item in df.iterrows():
                    denom = item[1][4]
                    for i, value in enumerate(item[1]):
                        if i not in range(5,10):
                            df.at[item[0],df.columns[i]] = value/denom
                        else:
                            df.at[item[0],df.columns[i]] = 0

                df.drop(['Unit (portion size)'],axis = 1, inplace=True)


                item_sim = cosine_similarity(normalize(df, axis=0))
                self.item_sim = item_sim

            if as_df:
                df_out = pd.DataFrame(data = self.item_sim, index = self.item_mapping, columns = self.item_mapping)
                if write_out:
                    df_out.to_csv("data\\item_sim.csv")
                return df_out

            else:
                if write_out:
                    np.savetxt("data\\item_sim_np.csv", self.item_sim, delimiter=",")

            return self.item_sim

    def construct_RHat(self, df = True, write_out = False, OVERRIDE = -10000):
        # breakpoint()
        pref = self.pref.loc[:, self.pref.columns != 'user_id']
        pref = pref.replace(0,OVERRIDE)

        if not self.been_fit:
            return pref

        RHat = np.zeros(pref.shape)

        no_rating = pref.isna().to_numpy()
        pref = pref.to_numpy()


        self.sim_mat()
        # calcing RHat
        for u,user in enumerate(RHat):
            for i, _ in enumerate(user):
                if no_rating[u][i]:
                    RHat[u][i] = self.estimate(u,i)
                else:
                    RHat[u][i] = pref[u][i]

        if df:
            df = pd.DataFrame(data = RHat, columns = ["oranges", "apples", "watermelon", "bananas", "eggplant", "tomatoes", "potatoes", "bread", "oats",
                                                        "rice", "fish", "eggs", "chicken", "olive_oil", "soybean", "beef", "milk", "yogurt", "cheese_ball"])
            if write_out:
                df.to_csv("data\\user_preferences.csv")

            return df
        else:
            if write_out:
                np.savetxt("data\\user_preferences_np.csv", RHat, delimiter=",")

        return RHat

    def add_new_user(self, new_user):
        self.data = self.data.append(new_user)
        self.user_sim = None

        pref_columns = ["user_id", "pref_oranges", "pref_apples", "pref_watermelon", "pref_bananas", "pref_eggplant", "pref_tomatoes",
                        "pref_potatoes", "pref_bread", "pref_oats", "pref_rice", "pref_fish", "pref_eggs", "pref_chicken", "pref_olive_oil",
                        "pref_soybean", "pref_beef", "pref_milk", "pref_yogurt", "pref_cheese_ball"]
        new_user_pref = new_user[pref_columns]
        new_user_pref["user_id"] = self.pref.index[-1]+1

        self.pref = self.pref.append(new_user_pref)

    def add_new_item(self, path):
        ## TODO:  WIP
        df=pd.read_csv(path).set_index('Item_name')
        f_groups=df['Food_group'].unique()
        temp=np.arange(1,f_groups.shape[0]+1)
        df['Food_group'] = df['Food_group'].replace(f_groups,temp)
        df.drop(['Notes/comments'],axis=1,inplace=True)



def nan_cosine_similarity(X, Y):
    if not len(X) == len(Y):
        raise ValueError("Input arrays must be the same length")
    else:
        mat = np.column_stack((X,Y)).transpose().astype(float)
        mat = mat[:, ~np.isnan(mat).any(axis=0)]
        return cosine_similarity(mat)[0][1]




def dataset(path, OVERRIDE = 0):
    user_responses = pd.read_csv(path)

    columns = ["user_id", "age_range", "white", "african_american", "asian", "native_american", "hawaiian", "other_race", "female", "male",
               "transgender", "other_gender", "income", "education", "house_size", "height", "weight", "allergy_none", "allergy_soybeans",
               "allergy_peanuts", "allergy_dairy", "allergy_wheat", "allergy_eggs", "allergy_fish", "allergy_shellfish", "allergy_treenuts",
               "allergy_meat", "allergy_other", "diet_none", "diet_veggie", "diet_vegan", "diet_kosher", "diet_beef", "diet_halal", "diet_red_meat",
               "diet_diabetic", "diet_gluten", "diet_other", "pref_orange", "pref_apple", "pref_watermelon", "pref_banana", "pref_eggplant",
               "pref_tomatoes", "pref_potatoes", "pref_bread", "pref_oats", "pref_rice", "pref_fish", "pref_egg", "pref_chicken", "pref_olive_oil",
               "pref_soybean_paste", "pref_beef", "pref_milk", "pref_yogurt", "pref_cheese_ball"]

    parsed_responses = pd.DataFrame(index= np.arange(0,len(user_responses)), columns = columns)

    # This is the map from given id and assigned id
    user_ids = user_responses['Please enter an identifier (ex. computing id)']

    user_responses = user_responses.drop(columns=['Timestamp', 'Please enter an identifier (ex. computing id)'])

    for user in user_responses.iterrows():
        # Age
        age_range = 1           if user[1][0] == "20-65" else 0 if user[1][0] == "2-19" else 2

        #Race/Ethnicity
        white = 1               if user[1][1] == "White" else 0
        african_american = 1    if user[1][1] == "Black or African American" else 0
        asian = 1               if user[1][1] == "Asian" else 0
        indian = 1              if user[1][1] == "American Indian or Alaska Native" else 0
        hawaiian = 1            if user[1][1] == "Native Hawaiian or Pacific Islander" else 0
        race_other = 1          if user[1][1] == "Other" else 0

        # Gender
        female = 1              if user[1][2] == "Female" else 0
        male = 1                if user[1][2] == "Male" else 0
        transgender = 1         if user[1][2] == "Transgender" else 0
        gender_other = 1        if user[1][2] == "Other" else 0

        # Income
        income_range = 1        if user[1][3] == "$23.6k-38.3k" else 0 if user[1][3] == "$23.6k or less" else 2

        # Education
        if user[1][4] == "Under High School":
            education = 0
        elif user[1][4] == "High School /GED":
            education = 1
        elif user[1][4] == "Associate Degree":
            education = 2
        elif user[1][4] == "Bachelor Degree":
            education = 3
        elif user[1][4] == "Graduate Degree":
            education = 4
        elif user[1][4] == "Masters Degree":
            education = 5
        elif user[1][4] == "Doctoral Degree":
            education = 6
        else:             #"Professional Degree"
            education = 7

        # Household size
        household = int(str(user[1][5])[0]) - 1

        #Body Characteristics
        height = float(user[1][6])
        weight = float(user[1][7])

        # Allergies
        allergies = user[1][8].split(', ')
        if allergies[0] == "No allergies":
            no_allergies = 1
            eggs = 0
            fish = 0
            dairy = 0
            wheat = 0
            peanuts = 0
            redmeat = 0
            soybeans = 0
            treenuts = 0
            shellfish = 0
            allergy_other = 0
        else:
            no_allergies = 0
            eggs = 1 if "Eggs" in allergies else 0
            fish = 1 if "Fish" in allergies else 0
            dairy = 1 if "Dairy" in allergies else 0
            wheat = 1 if "Wheat" in allergies else 0
            peanuts = 1 if "peanuts" in allergies else 0
            redmeat = 1 if "Red meat" in allergies else 0
            soybeans = 1 if "Soybeans" in allergies else 0
            treenuts = 1 if "Tree nuts" in allergies else 0
            shellfish = 1 if "Shellfish" in allergies else 0

            combined = soybeans + peanuts + dairy + wheat + eggs + fish + shellfish + treenuts + redmeat
            allergy_other = 1 if len(allergies) - combined > 0 else 0

        # Dietary Restrictions
        restrictions = user[1][9].split(', ')
        if restrictions[0] == "No restriction":
            no_restrictions = 1
            beef = 0
            vegan = 0
            halal = 0
            veggie = 0
            kosher = 0
            red_meat = 0
            diabetic = 0
            gluten_free = 0

            diet_other = 0
        else:
            no_restrictions = 0
            beef = 1 if "Beef" in restrictions else 0
            vegan = 1 if "Vegan" in restrictions else 0
            halal = 1 if "Halal" in restrictions else 0
            veggie = 1 if "Vegetarian" in restrictions else 0
            kosher = 1 if "Kosher" in restrictions else 0
            red_meat = 1 if "Red meat" in restrictions else 0
            diabetic = 1 if "Diabetic" in restrictions else 0
            gluten_free = 1 if "Gluten Free" in restrictions else 0

            combined = veggie + vegan + kosher + beef + halal + red_meat + diabetic + gluten_free + red_meat
            diet_other = 1 if len(restrictions) - combined > 0 else 0

        # User preferences
        pref_orange = user[1][10]
        pref_apple = user[1][11]
        pref_watermelon = user[1][12]
        pref_banana = user[1][13]
        pref_eggplant = user[1][14]
        pref_tomatoes = user[1][15]
        pref_potatoes = user[1][16]
        pref_bread = user[1][17]
        pref_oats = user[1][18]
        pref_rice = user[1][19]
        pref_fish = user[1][20]
        pref_egg = user[1][21]
        pref_chicken = user[1][22]
        pref_olive_oil = user[1][23]
        pref_soybean_paste = user[1][24]
        pref_beef = user[1][25]
        pref_milk = user[1][26]
        pref_yogurt = user[1][27]
        pref_cheese_ball = user[1][28]


        # Adjusting Preferences according to dietary restrictions
        if(vegan):
            pref_egg = OVERRIDE
            pref_milk = OVERRIDE
            pref_beef  = OVERRIDE
            pref_yogurt  = OVERRIDE
            pref_chicken = OVERRIDE
            pref_fish  = OVERRIDE
            pref_cheese_ball = OVERRIDE
        if(veggie):
            pref_beef  = OVERRIDE
            pref_chicken = OVERRIDE
            pref_fish  = OVERRIDE
        if(kosher):
            pass
        if(beef):
            pref_beef = OVERRIDE
        if(halal):
            pass
        if(red_meat):
            pref_beef = OVERRIDE
        if(diabetic):
            pref_bread = OVERRIDE
            pref_rice = OVERRIDE
        if(gluten_free):
            pref_bread = OVERRIDE

        # Adjusting preferences according to alergic restrictions
        if(eggs):
            pref_egg = OVERRIDE
        if(fish):
            pref_fish = OVERRIDE
        if(dairy):
            pref_milk = OVERRIDE
            pref_cheese_ball = OVERRIDE
            pref_yogurt = OVERRIDE
        if(wheat):
            pref_bread = OVERRIDE
        if(peanuts):
            pass
        if(redmeat):
            pref_beef = OVERRIDE
        if(soybeans):
            pref_soybean_paste = OVERRIDE
        if(treenuts):
            pass
        if(shellfish):
            pass


        parsed_responses.loc[user[0]] = [user[0], age_range, white, african_american, asian, indian, hawaiian, race_other, female, male, transgender,
                                         gender_other, income_range, education, household, height, weight, no_allergies, soybeans, peanuts, dairy,
                                         wheat, eggs, fish, shellfish, treenuts, redmeat, allergy_other, no_restrictions, veggie, vegan, kosher, beef,
                                         halal, red_meat, diabetic, gluten_free, diet_other, pref_orange, pref_apple, pref_watermelon, pref_banana,
                                         pref_eggplant, pref_tomatoes, pref_potatoes, pref_bread, pref_oats, pref_rice, pref_fish, pref_egg,
                                         pref_chicken, pref_olive_oil, pref_soybean_paste, pref_beef, pref_milk, pref_yogurt, pref_cheese_ball]


    return parsed_responses, user_ids


def main():
    #matrix rows are user and columns are items

    dataset()

    pref_matrix = construct_R()
    weights_matrix = np.random.rand(pref_matrix.shape[1],pref_matrix.shape[1])

    pred_pref_matrix = construct_RHat(pref_matrix, weights_matrix)


if __name__ == '__main__':
    main()
