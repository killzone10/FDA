import pandas as pd
import importlib.util
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# import train_predict from your file (change file name/path; omit suffix, i.e. .py)
m = importlib.import_module('Krajewski_B_img')

train_predict = m.__dict__['train_predict']

# load data
X = pd.read_csv("X_train.csv")
y = pd.read_csv("y_train.csv")

# split data (note: we use a different test set for grading purposes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=44)

# train and predict with function from student
y_pred = train_predict(X_train, y_train, X_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy: {}".format(acc))
