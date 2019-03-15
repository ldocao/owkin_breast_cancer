import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold


class PatientPredictor:
    N_FOLD = 5

    def cross_validation(self, x, y, seed=0):
        cls = self.__class__
        cv = StratifiedKFold(n_splits=cls.N_FOLD,
                             shuffle=True,
                             random_state=seed)
        auc = cross_val_score(self.estimator,
                              X=x, y=y,
                              cv=cv, scoring="roc_auc", verbose=0)
        return auc
    
class LogisticRegressionL2(PatientPredictor):

    def __init__(self):
        PARAMS = {"penalty": "l2",
                  "C": 0.8,
                  "solver": "liblinear"}
        self.estimator = LogisticRegression(**PARAMS)

    def train(self, x, y):
        self.estimator.fit(x, y)
        return self.estimator

    def predict(self, x):
        return self.estimator.predict_proba(x)
        
    def grid_search(self, x, y):
        cls = self.__class__
        GRID = {"penalty": ["l2"],
                "C": np.linspace(0.7, 1, 10),
                "solver": ["liblinear"]}
        grid_search = GridSearchCV(self.estimator, GRID, n_jobs=-1, cv=cls.N_FOLD)
        grid_search.fit(x, y)
        print("best score", grid_search.best_score_)
        return grid_search.best_params_
