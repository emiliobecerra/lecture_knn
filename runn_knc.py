import pandas
from sklearn.neighbors import KNeighborsClassifier
#Regressor for continuous, 
#Classifier for dummy, categorical variables
import kfold_template
dataset = pandas.read_csv("abalone.csv")
dataset = dataset.sample(frac=1)
# print(dataset)

#Predict the age group of the abalone, not the rings
dataset['Age Group'] = pandas.cut(dataset['Rings'], [0, 9, 11, 13, 100], labels=[1,2,3,4])
dataset = dataset.drop(['Rings'], axis=1)
# all the rows, column 8
target = dataset.iloc[:,8]
# we want the age of the abalone
# target = target + 1.5
target = target.values

data = dataset.iloc[:,0:8]
# We need to turn Sex into a 0,1,2 dummie variable
pandas.get_dummies(data, columns=['Sex'])
data = data.values
# print(data)

#Use a weighted average based on the distance
#Test the accuracy of all possible K's, uniform and distance
trials = []
for w in ['uniform', 'distance']:

	for k in range(1,50):
		machine = KNeighborsClassifier(n_neighbors=k, weights='distance')

#4 rounds, false for categorical 

		return_values = kfold_template.run_kfold(machine, data, target, 4, False)
		average_return_value = sum(return_values/len(return_values))
# print(return_values)
# print(average_return_value)
		trials.append((average_return_value, k, w))
#print(trials)

#We want the program to tell us which is the best trial.
#It's going to judge based on the first element, which is the average return value
#Lambda is a nameless function.
trials.sort(key=lambda x: x[0], reverse=True)
#Now it's sorted from the highest score to the lowest score. 

#Prints the best 5
print(trials[:5])
