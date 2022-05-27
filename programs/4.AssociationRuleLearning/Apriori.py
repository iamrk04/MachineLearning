import numpy as np
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv("4.AssociationRuleLearning/Market_Basket_Optimisation.csv", header=None)
transactions = []
for i in range(7501):
    transactions.append([str(dataset.values[i, j]) for j in range(20)])

# Training the Apriori model on the dataset
# min_support    = (min 3 times occurence a day) * (no. of days) / (total sample size) = 3 * 7 / 7500 = 0.003
# min_confidence = start with 0.8, if less rules in output, keep on decreasing it by dividing it by 2
# min_lift       = 3 (below 3, useless)
# min_length     = min items in the rules (including LHS and RHS), RHS will always have 1
# max_length     = max items in the rules (including LHS and RHS), RHS will always have 1
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

# Visualising the results
def inspect(results):
    lhs         = [result[2][0][0] for result in results]
    rhs         = [result[2][0][1] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

results = list(rules)
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

# Displaying the results non sorted
print(resultsinDataFrame)

print("-------------------------------------------------------------------------")

# Displaying the results sorted by descending lifts
print(resultsinDataFrame.nlargest(n = 10, columns = 'Lift'))