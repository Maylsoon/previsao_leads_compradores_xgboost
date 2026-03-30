from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

def train_baseline_model(X_train, y_train):
    
    # modelo
    model = LogisticRegression(max_iter=1000, random_state=42)
    
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
    
    # prints (mantendo seu estilo)
    print(f"ROC AUC em cada subconjunto:\n{scores.round(2)}")
    print(f"Média ROC AUC: {scores.mean():.4f}")
    print(f"Desvio padrão: {scores.std():.4f}")
    
    return model, scores