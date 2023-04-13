import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import csv
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


def submission(L):
    test = open("test.csv", "r")
    with open("submission.csv", "w") as f:
        writer = csv.writer(f)
        reader = csv.reader(test)

        writer.writerow(["url", "status"])
        for i, row in enumerate(reader):
            if i == 0:
                continue
            var = "phishing" if L[i - 1] == 1 else "legitimate"
            writer.writerow([row[1], var])
    test.close()


def build_submit_set(path):
    df = pd.read_csv(path, header=0)
    x = df._get_numeric_data()
    x = x.drop(columns=['Unnamed: 0'])
    x = x.to_numpy()
    x = normalize(x, axis=0)
    return x


def filter(L):
    res = []
    for i in L:
        if i > 0.5:
            res.append(1)
        else:
            res.append(0)
    return res


def compute_accuracy(Y_true, Y_pred):
    correctly_predicted = 0
    # iterating over every label and checking it with the true sample
    for true_label, predicted in zip(Y_true, Y_pred):
        if true_label == predicted:
            correctly_predicted += 1
    # computing the accuracy score
    accuracy_score = correctly_predicted / len(Y_true)
    return accuracy_score


def build_set(path):
    df = pd.read_csv(path, header=0)
    x = df._get_numeric_data()
    x = x.drop(columns=['Unnamed: 0'])
    x = x.to_numpy()
    x = normalize(x, axis=0)
    y = df["status"].map({'legitimate': 0, 'phishing': 1})
    # f = plt.figure(figsize=(25, 20))
    # plt.matshow(df.corr(), fignum=f.number)
    # plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=12,
    #            rotation=45)
    # plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=12)
    # cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=14)
    # plt.title('Correlation Matrix', fontsize=16)
    # plt.show()
    return x, y


# Define Grid
x_train, y_train = build_set("train.csv")
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size=0.3, shuffle=True)

grid = {
    'n_estimators': [1000, 1200, 1400, 1600],
    'max_features': ['sqrt'],
    'max_depth': [24, 28, 30, 32],
    'random_state': [18]
}
# show start time
print(datetime.now())  # Grid Search function

# CV_rfr = GridSearchCV(estimator=RandomForestClassifier(),
#                       param_grid=grid, cv=5, verbose=10, n_jobs=-1)
# CV_rfr.fit(x_train, y_train)

# show end time
print(datetime.now())
# print("\n The best parameters across ALL searched params:\n", CV_rfr.best_params_)

rf = RandomForestClassifier(n_estimators=1000, max_features='sqrt', max_depth=28, random_state=18)

rf.fit(x_train, y_train)
# print(rf.predict(x_test))

x_test_submit = build_submit_set("test.csv")
prediction = filter(rf.predict(x_test_submit))
# print("Accuracy: ", compute_accuracy(y_test, prediction))
submission(prediction)
