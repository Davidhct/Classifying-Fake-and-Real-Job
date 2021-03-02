import numpy as np
from preprocessing_data import DataProccess
import utils
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, confusion_matrix
from mlxtend.plotting import plot_learning_curves
from matplotlib import pyplot
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier



class Estimators:
    """
    This class holds all the Estimators from sk-learn to be used in the training

    SVC with 'rbf'
    Logistic Regression max iteration 2000
    KNN with 2 neighbors
    Random Forest to 3 estimators
    Adaboost


    """
    def __init__(self):
        self.svm = SVC(kernel='rbf') ## default 'rbf'
        self.logR = LogisticRegression(max_iter=2000)
        self.knn = KNeighborsClassifier(n_neighbors=2)
        self.rfc = RandomForestClassifier(n_estimators=3)
        self.ada = AdaBoostClassifier()

        self.estimators = {"SVM": self.svm,
                           "Logistic Regression":  self.logR,
                           "KNN":  self.knn,
                           "Random Forest":  self.rfc,
                           "Adaboost": self.ada}

def main():
    dp = DataProccess()
    es = Estimators()

    features_ub, labels_ub = dp.prepare_data()

    ros = RandomOverSampler()
    features_b, labels_b = ros.fit_resample(features_ub, labels_ub)

    X_train, X_test, y_train, y_test = train_test_split(features_b, labels_b, test_size=0.2)


    # print("X_train.shape = ",X_train.shape)
    # print("y_train.shape = ",y_train.shape)
    # print("X_test.shape = ",X_test.shape)
    # print("y_test.shape = ",y_test.shape)


    estimators = es.estimators
    comparison = {}
    for name in estimators:
        print(f"Training Model with Estimator: {name}")
        estimator = estimators[name]

        estimator.fit(X_train, y_train)

        # Predicting
        y_pred = estimator.predict(X_test)

        # Model Accuracy
        acc = accuracy_score(y_test, np.round(y_pred)) * 100
        data = {name: acc}
        comparison.update(data)

        cm = confusion_matrix(y_test, np.round(y_pred))
        tn, fp, fn, tp = cm.ravel()

        print('------------- CONFUSION MATRIX ----')
        print(cm)

        print('\n------------- TEST METRICS -------')
        precision = tp/(tp+fp)*100
        recall = tp/(tp+fn)*100
        print('Accuracy: {}%'.format(acc))
        print('Precision: {}%'.format(precision))
        print('Recall: {}%'.format(recall))
        print('F1-score: {}%'.format(2*precision*recall/(precision+recall)))

        plot_learning_curves(X_train, y_train, X_test, y_test, estimator)
        pyplot.ylim(bottom=-0.0010)
        pyplot.show()
        print('----------------------------------------------------------------')


    utils.accuracy_comparison(comparison)


if __name__ == '__main__':
    main()
