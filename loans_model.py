import os
import numpy as np    # For array operations
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd   # For DataFrames
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import plotly.express as px
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from sklearn.model_selection import LeaveOneOut, KFold


def randomForest(X_train, X_test, y_train, model):
    model.fit(X_train, y_train)
    return model.predict(X_test)

def decisionTree(X_train, X_test, y_train, model):
    model.fit(X_train, y_train)
    return model.predict(X_test)

def nn(X_train, X_test, y_train, model):
    model.add(Dense(64, input_shape=(12,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    '''es = EarlyStopping(
        patience=15,
        monitor="accuracy",
        restore_best_weights=True
    )'''

    model.fit(X_train, y_train, epochs=200, batch_size=8)

    predictions = model.predict(X_test)
    test_predictions = [1 if num > 0.5 else 0 for num in predictions]
    return test_predictions

if __name__ == '__main__':
    loans = pd.read_csv('./loan_database.csv')

    X = loans.copy().drop("DECISION", axis=1)
    y = loans["DECISION"]

    #oversample = RandomOverSampler(sampling_strategy="minority")
    #oversample = SMOTE()
    #X, y = oversample.fit_resample(X, y)
    cv = KFold(n_splits=5, shuffle=True)

    y_true = list()
    y_pred_rf = list()
    y_pred_dt = list()
    y_pred_nn = list()

    for i,(train_ix, test_ix) in enumerate(cv.split(X)):
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]  # Use iloc to access data using indices
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]  # Use iloc to access data using indices

        scaler = MinMaxScaler()
        #scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        yhat_rf = randomForest(X_train, X_test, y_train, RandomForestClassifier(random_state=42, n_estimators=100))
        yhat_dt = decisionTree(X_train, X_test, y_train, DecisionTreeClassifier(random_state=42))
        yhat_nn = nn(X_train, X_test, y_train, Sequential())

        y_true.append(y_test.tolist())
        y_pred_rf.append(yhat_rf.tolist())
        y_pred_dt.append(yhat_dt.tolist())
        y_pred_nn.append(yhat_nn)

    y_true = [item for sublist in y_true for item in sublist]
    y_pred_rf = [item for sublist in y_pred_rf for item in sublist]
    y_pred_dt = [item for sublist in y_pred_dt for item in sublist]
    y_pred_nn = [item for sublist in y_pred_nn for item in sublist]

    print("____________Random Forest______________")
    acc = accuracy_score(y_true, y_pred_rf)
    print('Accuracy: %.3f' % acc)
    print(classification_report(y_true, y_pred_rf))
    print(confusion_matrix(y_true, y_pred_rf))

    print("____________Decision Trees______________")
    acc = accuracy_score(y_true, y_pred_dt)
    print('Accuracy: %.3f' % acc)
    print(classification_report(y_true, y_pred_dt))
    print(confusion_matrix(y_true, y_pred_dt))

    print("____________Neural Network______________")
    acc = accuracy_score(y_true, y_pred_nn)
    print('Accuracy: %.3f' % acc)
    print(classification_report(y_true, y_pred_nn))
    print(confusion_matrix(y_true, y_pred_nn))

    res_pred = []

    for i in range(len(y)):
        if y_pred_rf[i] + y_pred_dt[i] + y_pred_nn[i] > 1:
            res_pred.append(1)
        else:
            res_pred.append(0)
    print("____________Skupaj______________")
    acc = accuracy_score(y_true, res_pred)
    print('Accuracy: %.3f' % acc)
    print(classification_report(y_true, res_pred))
    print(confusion_matrix(y_true, res_pred))
