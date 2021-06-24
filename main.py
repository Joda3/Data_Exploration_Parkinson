import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif

import seaborn as sea

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import optuna


# import the dataset (change local path here)
data = pd.read_csv('C:/Users/Johannes/Desktop/DHBW/4. Semester/data expo/parkinsons2.csv', delimiter=",")

# check if the dataset is correctly imported
print(data.head())

#check if some values are empty
print(data.isnull().sum())

# plot a heatmap to see correlated attributes
sea.heatmap(data.corr(), cmap='inferno_r').set_title("correlated attributes")

# create a new dataset without attributes status and name
data_wo_name_status = data.drop(['status', 'name'], axis=1)
# create a new dataset with only status
data_status = data['status']


# plot of the correlating attributes to status
mutual_info = mutual_info_classif(data_wo_name_status, data_status)
figure(figsize=(28, 6), dpi=80)
sea.barplot(data_wo_name_status.columns, mutual_info, palette='magma').set_title("correlating attributes to status")


# create a boxplot for each attribute except status and name to get more detailed information about the dataset
for attribute in data_wo_name_status.columns:
    plt.figure()
    sea.boxplot(x="status", y=attribute, data=data, palette='magma').set_title("Boxplot")


# Scaling to normalize the dataset
# create a Scaler
scaler = StandardScaler() # you can change the Scaler here
pd.DataFrame(data_wo_name_status)
scaled_data = scaler.fit_transform(data_wo_name_status)
scaled_data = pd.DataFrame(scaled_data)

# print the normalized data
print(scaled_data)


# split the dataset into train and test sets
a_train, a_test, b_train, b_test = train_test_split(scaled_data, data_status,
                                                    test_size=0.25, shuffle=True, stratify=data_status, random_state=0)

# use DecisionTree classifier
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(a_train, b_train)
print("Mean accuracy for the training set: ", round(classifier.score(a_train, b_train), 5))
print("Mean accuracy for the test set: ", round(classifier.score(a_test, b_test), 5))
# plot confusion matrix for the train set
plot_confusion_matrix(classifier, a_train, b_train, normalize='true', cmap='Purples',
                      display_labels=["Healthy", "Parkinson"]).ax_.set_title("Training set (first run)")


# plot confusion matrix for test set
plot_confusion_matrix(classifier, a_test, b_test, normalize='true', cmap='Purples',
                      display_labels=["Healthy", "Parkinson"]).ax_.set_title("Test set (first run)")


# define the input variables for the decision tree
def decisiontree(trial):
    opt_criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    opt_splitter = trial.suggest_categorical("splitter", ["random", "best"])
    opt_max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
    opt_min_samples_split = trial.suggest_int("min_samples_split", 2, 20, log=True)
    # initialize the DecisionTree
    classifier_obj = DecisionTreeClassifier(criterion=opt_criterion,
                                            splitter=opt_splitter,
                                            max_depth=opt_max_depth,
                                            min_samples_split=opt_min_samples_split,
                                            random_state=0,
                                            max_features=None)
    # calculate the accuracy
    score = cross_val_score(classifier_obj, scaled_data, data_status, n_jobs=-1, cv=10)
    accuracy = score.mean()
    return accuracy


# run the decision tree 100 times
opt_study = optuna.create_study(direction="maximize")
opt_study.optimize(decisiontree, n_trials=100)
print("This is the best trial: ", opt_study.best_trial)
# print the best parameters found
print("Those are the best params: ", opt_study.best_params)


# initialize the DecisionTree again with the best parameters found
classifier = DecisionTreeClassifier(
    criterion=opt_study.best_params['criterion'],
    splitter=opt_study.best_params['splitter'],
    max_depth=opt_study.best_params['max_depth'],
    min_samples_split=opt_study.best_params['min_samples_split'],
    random_state=0,
    max_features=None)

# input the training samples
classifier.fit(a_train, b_train)
# print the accuracy of the train and test sets
print("Mean accuracy for the training set after optuna : ", round(classifier.score(a_train, b_train), 5))
print("Mean accuracy for the test set after optuna: ", round(classifier.score(a_test, b_test), 5))


classifier = DecisionTreeClassifier(
    criterion='gini',
    splitter='random',
    max_depth=6,
    min_samples_split=2,
    random_state=0)
classifier.fit(a_train, b_train)

print("----------------------------------------------------------------")
print("Mean accuracy for the training set with the best known params: ", round(classifier.score(a_train, b_train), 5))
print("Mean accuracy for the test set with the best known params: ", round(classifier.score(a_test, b_test), 5))

# plot confusion matrix for train set one more time
plot_confusion_matrix(classifier, a_train, b_train,
                      normalize='true',
                      cmap='Purples',
                      display_labels=["Healthy", "Parkinson"]).ax_.set_title("Training set with best params")

# plot confusion matrix for test set one more time
plot_confusion_matrix(classifier, a_test, b_test,
                      normalize='true',
                      cmap='Purples',
                      display_labels=["Healthy", "Parkinson"]).ax_.set_title("Test set with best params")

# show all plots
plt.show()
