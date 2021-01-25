import sklearn
from sklearn import svm
from sklearn import datasets
from sklearn import metrics
import pickle

# load data
cancer = datasets.load_breast_cancer()
# see data features
#print("Features: ", cancer.feature_names)
# see data labels
#print("Labels: ", cancer.target_names)

# splitting data
x = cancer.data  # All of the features
y = cancer.target  # All of the labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
print(x_train, y_train)

'''
# training multiple models to get and save(using Pickle) the best model
best = 0
for i in range(10):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

    # Training SVM Classifier
    clf = svm.SVC(kernel="linear", C=2)
    clf.fit(x_train, y_train)

    # Predictions
    y_pred = clf.predict(x_test)

    # accuracy
    acc = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy: " + str(acc))

    # If the current model has a better score than one we've already trained then save it
    if acc > best:
        best = acc
        with open("SVM-model.pickle", "wb") as f:
            pickle.dump(clf, f)
'''
# loading model
pickle_in = open("SVM-model.pickle", "rb")
model = pickle.load(pickle_in)

# Testing model
predicted = model.predict(x_test)
names = ["malignant", "benign"]

# This will display the predicted class, our data and the actual class
# We create a names list so that we can convert our integer predictions into
# their string representation
for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "\tActual: ", names[y_test[x]]) # if you want to see data add this "Data: ", x_test[x],
