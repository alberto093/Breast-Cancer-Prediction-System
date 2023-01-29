# Breast Cancer Prediction System
# Create by Alberto Saltarelli on 25/01/2023.
# Copyright © 2023 BCPS - Alberto Saltarelli

import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pgmpy

from numpy import mean
from numpy import std
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from pgmpy.estimators import K2Score, HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from sklearn import svm
from sklearn.svm import SVC


class ClassifierModel:
    accuracy_list = 0.0
    precision_list = 0.0
    recall_list = 0.0
    f1_list = 0.0

Cancer = pd.read_csv("data.csv")

print(
    "\n\nBreast Cancer Prediction System - Il sistema predice se, presi dei soggetti, essi sono affetti o meno dal cancro al seno\n")
print("Il dataset utilizzato è il Breast Cancer Prediction | Kaggle", Cancer)

# Split the dataset into input features and target feature
X = Cancer.drop("diagnosis", axis=1)
Y = Cancer["diagnosis"]

# Balancing dataset
Positive = Cancer.diagnosis.value_counts()[1]
Negative = Cancer.diagnosis.value_counts()[0]
print("Pazienti malati: ", Positive, "({:.2f}%)".format(Positive / Cancer.diagnosis.count() * 100))
print("Pazienti sani: ", Negative, "({:.2f}%)".format(Negative / Cancer.diagnosis.count() * 100), "\n")

# Plot visualization
chart = Cancer["diagnosis"].value_counts().plot(kind="bar", rot=0)
chart.set_title("Cancer Breast Wisconsin Data Set")
chart.set_xticklabels(["Sani", "Malati"])
plt.show()

X = Cancer.to_numpy()
Y = Cancer["diagnosis"].to_numpy()

# K-Fold Cross Validation
kf = StratifiedKFold(n_splits=5)  # La classe è in squilibrio, quindi utilizzo Stratified K-Fold
print("È stato applicato un K-Fold Cross Validation con k = 5 per ottenere un dataset bilanciato\n")

# Classifier comparison
knn = KNeighborsClassifier()
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
svc = SVC()

# Score
knn_model = ClassifierModel()
dtc_model = ClassifierModel()
rfc_model = ClassifierModel()
svc_model = ClassifierModel()

# K-Fold Scoring
for train_index, test_index in kf.split(X, Y):
    training_set = X[train_index]
    test_set = X[test_index]

    # Training Set Data
    data_train = pd.DataFrame(training_set, columns=Cancer.columns)
    X_train = data_train.drop("diagnosis", axis=1)
    Y_train = data_train.diagnosis

    # Test Set Data
    data_test = pd.DataFrame(test_set, columns=Cancer.columns)
    X_test = data_test.drop("diagnosis", axis=1)
    Y_test = data_test.diagnosis

    # Classifier Fitting (Learner)
    knn.fit(X_train, Y_train)
    dtc.fit(X_train, Y_train)
    rfc.fit(X_train, Y_train)
    svc.fit(X_train, Y_train)

    # Inference Model
    Y_knn = knn.predict(X_test)
    Y_dtc = dtc.predict(X_test)
    Y_rfc = rfc.predict(X_test)
    Y_svc = svc.predict(X_test)

    knn_model.accuracy_list = (metrics.accuracy_score(Y_test, Y_knn))
    knn_model.precision_list = (metrics.precision_score(Y_test, Y_knn))
    knn_model.recall_list = (metrics.recall_score(Y_test, Y_knn))
    knn_model.f1_list = (metrics.f1_score(Y_test, Y_knn))

    dtc_model.accuracy_list = (metrics.accuracy_score(Y_test, Y_dtc))
    dtc_model.precision_list = (metrics.precision_score(Y_test, Y_dtc))
    dtc_model.recall_list = (metrics.recall_score(Y_test, Y_dtc))
    dtc_model.f1_list = (metrics.f1_score(Y_test, Y_dtc))

    rfc_model.accuracy_list = (metrics.accuracy_score(Y_test, Y_rfc))
    rfc_model.precision_list = (metrics.precision_score(Y_test, Y_rfc))
    rfc_model.recall_list = (metrics.recall_score(Y_test, Y_rfc))
    rfc_model.f1_list = (metrics.f1_score(Y_test, Y_rfc))

    svc_model.accuracy_list = (metrics.accuracy_score(Y_test, Y_svc))
    svc_model.precision_list = (metrics.precision_score(Y_test, Y_svc))
    svc_model.recall_list = (metrics.recall_score(Y_test, Y_svc))
    svc_model.f1_list = (metrics.f1_score(Y_test, Y_svc))

# Visualization of data frame of metrics
classifier_data = {"Classifier": ["KNN", "Decision Tree", "Random Forest", "SVM"],
                   "Accuracy": [np.mean(knn_model.accuracy_list), np.mean(dtc_model.accuracy_list),
                                np.mean(rfc_model.accuracy_list), np.mean(svc_model.accuracy_list)],
                   "Precision": [np.mean(knn_model.precision_list), np.mean(dtc_model.precision_list),
                                 np.mean(rfc_model.precision_list), np.mean(svc_model.precision_list)],
                   "Recall": [np.mean(knn_model.recall_list), np.mean(dtc_model.recall_list),
                              np.mean(rfc_model.recall_list), np.mean(svc_model.recall_list)],
                   "F1 Score": [np.mean(knn_model.f1_list), np.mean(dtc_model.f1_list), np.mean(rfc_model.f1_list),
                                np.mean(svc_model.f1_list)],
                   }

data_frame = pd.DataFrame(classifier_data)
data_frame.set_index("Classifier", inplace=True)
print(data_frame)

# Feature Selection
X = Cancer.drop("diagnosis", axis=1)
Y = Cancer["diagnosis"]

rfc = RandomForestClassifier(random_state=42, n_estimators=100)  # 42 means shuffled data set
rfc_model = rfc.fit(X, Y)

chart = (pd.Series(rfc_model.feature_importances_, index=X.columns) \
         .nlargest(10)  # limit to 10 features
         .plot(kind="barh", figsize=[14, 7]))

chart.set_title("Random Forest - Feature Selection")
chart.invert_yaxis()  # descending order
plt.show()

## Bayesian network

# Integer conversion for testing/performance purposes
Cancer = pd.DataFrame(np.array(Cancer, dtype=int), columns=Cancer.columns).iloc[:-2,
         :]  # Last two elements will be used for queries

# Network model
k2_model = HillClimbSearch(Cancer).estimate(scoring_method=K2Score(Cancer))

# Network creation
network = BayesianNetwork(k2_model.edges())
network.fit(Cancer, estimator=MaximumLikelihoodEstimator)

# Network visualization
Graph = nx.DiGraph()
Graph.add_nodes_from(network.nodes)
Graph.add_edges_from(network.edges)
nx.draw(Graph, pos=nx.spring_layout(Graph, k=12 / np.sqrt(Graph.order())), with_labels=True)
plt.figure(1, figsize=(12, 12))
plt.show()

# Probability Calculus

# Variable elimination
data = VariableElimination(network)

# Calculate the probabilty of breast cancer about healthy (0) and cancer (1) patients.


# Potentially healthy patient
healthy = data.query(variables=["diagnosis"],
                    evidence={"radius_mean": 8, "texture_mean": 25, "perimeter_mean": 48, "area_mean": 181,
                              "radius_se": 0, "perimeter_se": 3, "area_se": 19, "radius_worst": 9,
                              "texture_worst": 30, "perimeter_worst": 59, "area_worst": 269,
                              "compactness_worst": 0, "concavity_worst": 0})

print("\nProbabilità della positività per un soggetto potenzialmente sano:")
print(healthy)

# Potentially cancer patient
cancer = data.query(variables=["diagnosis"],
                    evidence={"radius_mean": 21, "texture_mean": 29, "perimeter_mean": 140, "area_mean": 1265,
                              "radius_se": 726, "perimeter_se": 6, "area_se": 86, "radius_worst": 26,
                              "texture_worst": 39, "perimeter_worst": 185, "area_worst": 1821,
                              "compactness_worst": 1, "concavity_worst": 1})

print("\nProbabilità per un soggetto potenzialmente malato:")
print(cancer)