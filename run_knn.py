import pandas
from sklearn.neighbors import KNeighborsRegressor
import kfold_template
dataset = pandas.read_csv("abalone.csv")
dataset = dataset.sample(frac=1)
# print(dataset)

# all the rows, column 8
target = dataset.iloc[:,8]
# we want the age of the abalone
target = target + 1.5
target = target.values

data = dataset.iloc[:,0:8]
# We need to turn Sex into a 0,1,2 dummie variable
pandas.get_dummies(data, columns=['Sex'])
data = data.values
# print(data)

#Use a weighted average based on the distance
machine = KNeighborsRegressor(n_neighbors=5, weights='uniform')

#4 rounds, true for continuous 
return_values = kfold_template.run_kfold(machine, data, target, 4, True)
average_return_value = sum(return_values/len(return_values))
print(return_values)