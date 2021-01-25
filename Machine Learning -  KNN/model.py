from sklearn import preprocessing
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import pickle

# load data
data = pd.read_csv("car.data")
print(data.head())

# convert string data into integers and separate data
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

# what we want to predict
predict = "class"
# Features
X = list(zip(buying, maint, door, persons, lug_boot, safety))
# Labels
y = list(cls)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

'''
# training multiple models to get and save(using Pickle) the best model
best = 0
for i in range(50):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    # Training KNN Classifier (specifying number of neighbours)
    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(X_train, y_train)
    # accuracy
    acc = model.score(X_test, y_test)
    print("Accuracy: " + str(acc))

    # If the current model has a better score than one we've already trained then save it
    if acc > best:
        best = acc
        with open("KNN-model.pickle", "wb") as f:
            pickle.dump(model, f)
'''

# loading model
pickle_in = open("KNN-model.pickle", "rb")
model = pickle.load(pickle_in)

# Testing model
predicted = model.predict(X_test)
names = ["unacc", "acc", "good", "vgood"]

# This will display the predicted class, our data and the actual class
# We create a names list so that we can convert our integer predictions into
# their string representation
for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", X_test[x], "Actual: ", names[y_test[x]])
    # Looking at neighbours
    n = model.kneighbors([X_test[x]], 9, True)
   # print("N: ", n)

