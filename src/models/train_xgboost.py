from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def train_xgboost_model(X_train, y_train):
    
    # modelo
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    
    # cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    
    # avaliação
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring='roc_auc'
    )
    
    print(f"ROC AUC em cada subconjunto:\n{scores.round(2)}")
    print(f"Média ROC AUC: {scores.mean():.4f}")
    print(f"Desvio padrão: {scores.std():.4f}")
    
    return model, scores