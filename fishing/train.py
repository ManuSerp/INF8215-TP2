import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
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
    selected_columns = ['length_url', 'length_hostname', "nb_dots", "nb_hyphens", "nb_at", "nb_qm", "nb_and", "nb_or",
                        "nb_eq", "nb_underscore", "nb_tilde", "nb_percent", "nb_slash", "nb_star", "nb_colon",
                        "nb_comma", "nb_semicolumn", "nb_dollar", "nb_space", "nb_www", "nb_com", "nb_dslash",
                        "ratio_digits_url", "ratio_digits_host", "punycode", "path_extension", "nb_redirection",
                        "nb_external_redirection",
                        "ratio_intRedirection", "ratio_extRedirection", "login_form", "external_favicon", "sfh",
                        "iframe", "popup_window", "safe_anchor", "onmouseover", "right_clic", "empty_title",
                        "domain_in_title", "domain_in_brand", "brand_in_subdomain", "brand_in_path",
                        "whois_registered_domain", "domain_registration_length", "domain_age", "web_traffic",
                        "dns_record", "google_index", "page_rank"]
    x = df[selected_columns]
    x = x._get_numeric_data()
    # x = x.drop(columns=['Unnamed: 0'])
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


def build_submit_set(path):
    df = pd.read_csv(path, header=0)
    x = df._get_numeric_data()
    x = x.drop(columns=['Unnamed: 0'])
    x = x.to_numpy()
    x = normalize(x, axis=0)
    return x


def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2), len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2), len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1, 1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx, :], '-o', label=name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    plt.show()


def build_set(path):
    df = pd.read_csv(path, header=0)
    selected_columns = ['length_url', 'length_hostname', "nb_dots", "nb_hyphens", "nb_at", "nb_qm", "nb_and", "nb_or",
                        "nb_eq", "nb_underscore", "nb_tilde", "nb_percent", "nb_slash", "nb_star", "nb_colon",
                        "nb_comma", "nb_semicolumn", "nb_dollar", "nb_space", "nb_www", "nb_com", "nb_dslash",
                        "ratio_digits_url", "ratio_digits_host", "punycode", "path_extension", "nb_redirection",
                        "nb_external_redirection",
                        "ratio_intRedirection", "ratio_extRedirection", "login_form", "external_favicon", "sfh",
                        "iframe", "popup_window", "safe_anchor", "onmouseover", "right_clic", "empty_title",
                        "domain_in_title", "domain_in_brand", "brand_in_subdomain", "brand_in_path",
                        "whois_registered_domain", "domain_registration_length", "domain_age", "web_traffic",
                        "dns_record", "google_index", "page_rank"]
    x = df[selected_columns]
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
    'n_estimators': [600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    'max_features': ['sqrt'],
    'max_depth': [20, 24, 28, 30, 32],
    'random_state': [18]
}
# show start time
print(datetime.now())  # Grid Search function

CV_rf = GridSearchCV(estimator=RandomForestClassifier(),
                     param_grid=grid, cv=5, verbose=10, n_jobs=-1)

# Calling Method

CV_rf.fit(x_train, y_train)
plot_grid_search(CV_rf.cv_results_, grid['n_estimators'], grid['max_depth'], 'N Estimators', 'Max Depth')

# show end time
print(datetime.now())
# print("\n The best parameters across ALL searched params:\n", rf.best_params_)

rf = RandomForestClassifier(n_estimators=CV_rf.best_params_["n_estimators"], max_features='sqrt',
                            max_depth=CV_rf.best_params_["max_depth"], random_state=CV_rf.best_params_["random_state"])
# rf = RandomForestClassifier(n_estimators=1200, max_features='sqrt', max_depth=24, random_state=18)

rf.fit(x_train, y_train)
print(rf.predict(x_test))
prediction = filter(rf.predict(x_test))
print("Accuracy: ", compute_accuracy(y_test, prediction))
print(confusion_matrix(filter(y_test), prediction))
x_test_submit = build_submit_set("test.csv")
# prediction = filter(rf.predict(x_test_submit))
# print("Accuracy: ", compute_accuracy(y_test, prediction))
# submission(prediction)
