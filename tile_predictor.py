import abc
import numpy as np

from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold


from camelyon16 import TrainingPatients

class TilePredictor:
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



    def predict_tiles_of(self, ids):
        """Returns positive probability for every tile of patient ids

        Parameters
        ----------
        ids: iterable
            list of patient ids
        
        Returns
        -------
        results: dict
            prediction positive probabilities for each patient
        """
        results = {}
        n = len(ids)
        count = 0
        for p in ids:
            print(count/n)
            resnet = TrainingPatients().resnet_features(p)
            n_tiles = resnet.shape[0]
            proba = self.estimator.predict_proba(resnet)[:,1]
            results[p] = proba
            count += 1
        return results

    @abc.abstractmethod
    def train(self, x, y):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def grid_search(self, x, y):
        raise NotImplementedError



    
class LogisticRegressionL2(TilePredictor):

    def __init__(self):
        PARAMS = {"penalty": "l2",
                  "C": 0.000774263682681127,
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
                "C": np.logspace(-4, -2, 10),
                "solver": ["liblinear"]}
        grid_search = GridSearchCV(self.estimator, GRID, n_jobs=-1, cv=cls.N_FOLD)
        grid_search.fit(x, y)
        print("best score", grid_search.best_score_)
        return grid_search.best_params_



class XGBoost(TilePredictor):
    def __init__(self):
        PARAMS = {'max_depth': 3,
                  'eta': 0.3,
                  'silent': 1,
                  'objective': 'binary:logistic',
                  'eval_metric': 'auc'}
        self.params = PARAMS
    
    def train(self, x, y):
        self.estimator.fit(x, y, metric='auc')
        return self.estimator

    def cross_validation(self, x, y):
        cls = self.__class__
        x_train = xgb.DMatrix(x, label=y)
        results = xgb.cv(self.params, x_train,
                         n_fold=cls.N_FOLD, num_boost_round=3,
                         stratified=True, shuffle=True)
        return results
                
