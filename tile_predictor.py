import abc
import numpy as np

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold




class TilePredictor:
    N_FOLD = 5
    
    @abc.abstractmethod
    def train(self, x, y):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, x):
        raise NotImplementedError()




    
class LogisticRegressionL2(TilePredictor):

    def __init__(self):
        PARAMS = {"penalty": "l2",
                  "C": 1.,
                  "solver": "liblinear"}
        self.estimator = LogisticRegression(**PARAMS)

    def train(self, x, y):
        self.estimator.fit(x, y)
        return estimator

    def predict(self, x):
        return self.estimator.predict_proba(x)
        

    def cross_validation(self, x, y, seed=0):
        cls = self.__class__
        cv = StratifiedKFold(n_splits=cls.N_FOLD,
                             shuffle=True,
                             random_state=seed)
        auc = cross_val_score(self.estimator,
                              X=x, y=y,
                              cv=cv, scoring="roc_auc", verbose=0)
        return auc



class XGBoost(TilePredictor):
    def __init__(self):
        PARAMS = {"n_jobs": -1,
                  "objective": "binary:logistic",
                  "max_depth": 6,
                  "learning_rate": 0.1}
        self.estimator = XGBClassifier(**PARAMS)

    def train(self, x, y):
        self.estimator.fit(x, y)
        return self.estimator

    def grid_search(self, x, y):
        GRID = {"max_depth": [int(f) for f in np.logspace(0, 2, 3)],
                "learning_rate": np.logspace(-3, -1, 3),
                "n_estimators": [int(f) for f in np.logspace(1, 2.5, 3)]}
        grid_search = GridSearchCV(self.estimator, GRID, cv=self.__class__.N_FOLD, n_jobs=-1)
        grid_search.fit(x, y)
        return grid_search.best_params_
        
    def cross_validation(self, x, y):
        clf.fit(train_cluster_x,train_cluster_y)
        clf.best_score_, clf.best_params_

        return auc

