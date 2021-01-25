import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import style

# load data
data = pd.read_csv("student-mat.csv", sep=";")

# trim data
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())

# separate data
# want to predict final grade (G3)
predict = "G3"
# Features
X = np.array(data.drop([predict], 1))
# Labels
y = np.array(data[predict])

# X train etc is redefined here so when training multiple models is commented out we still have these varibales to work with
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1) # test_size = 0.1 splits 10% of data into test samples
'''
# training multiple models to get and save(using Pickle) the best model
best = 0
for i in range(50):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc))

    # If the current model has a better score than one we've already trained then save it
    if acc > best:
        best = acc
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)
'''
# loading model
pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)

# viewing constants
print('Coefficient: \n', linear.coef_) # These are each slope value
print('Intercept: \n', linear.intercept_) # This is the intercept

# predicting on specific students
predictions = linear.predict(x_test) # Gets a list of all predictions
print("predictions")
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# output e.g.: 11.557670197820379 [12 11  1  0 16] 11
# first number is the predicted grade [beginning grade was 12, end of term grade was 11, 1 hour of studytime, 0 failures, 4 absences] their actual grade was 15

# plotting data - for visual representation
style.use("ggplot")
plot = "G2" # Change this to G1, G2, studytime, failures or absences to see other graphs
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()