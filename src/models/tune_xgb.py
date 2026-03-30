from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

def tune_xgboost_model(X_train, y_train):
    
    param_dist = {
        'n_estimators': [200, 400, 600],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 2, 5]
    }
    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    
    model = XGBClassifier(
        random_state=42,
        eval_metric='auc',
    )
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=20,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    print(f"Melhores parâmetros:\n{best_params}")
    
    return best_model, best_params