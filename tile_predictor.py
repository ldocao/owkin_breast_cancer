import abc

import xgboost as xgb
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
    def train(self, x, y):
        PARAMS = {"penalty": "l2",
                  "C": 1.,
                  "solver": "liblinear"}
        estimator = LogisticRegression(**PARAMS)
        estimator.fit(x, y)
        self.estimator = estimator #must possess predict_proba method
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
    def cross_validation(self, x, y):
        cls = self.__class__
        PARAM = {'max_depth': 5,
                 'eta': 0.5,
                 'silent': 1,
                 'objective': 'binary:logistic'}
        auc = xgb.cv(PARAM,
                     xgb.DMatrix(x, label=y),
                     metrics="auc",
                     nfold=cls.N_FOLD, stratified=True)
        return auc

