import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

eval = pd.read_csv("eval_day_1.csv")
eval = pd.DataFrame(eval)
eval = pd.melt(eval, id_vars=['ID'], value_vars=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                                                 '13', '14', '15', '16', '17', '18'],
               value_name="rating", var_name="itemID")
eval.rename(columns={"ID": "username"}, inplace=True)
print(eval)

pred = pd.read_csv("predicted_allocation_feedback_day_1.csv")
pred = pd.DataFrame(pred)
pred = pred[['username', 'itemID', 'rating']]


pred = pred.sort_values(by=['itemID'])
print(pred)

mse_1 = mean_squared_error(eval['rating'], pred['rating'])
print(sqrt(mse_1))