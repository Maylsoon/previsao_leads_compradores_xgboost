import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(X_train, X_test):
    
    # cópias (evita sobrescrever original)
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # =========================
    # TRANSFORMAÇÃO LOG
    # =========================
    
    X_train['AnnualSalary_log'] = np.log1p(X_train['AnnualSalary'])
    X_test['AnnualSalary_log'] = np.log1p(X_test['AnnualSalary'])
    
    # remover coluna original
    X_train = X_train.drop('AnnualSalary', axis=1)
    X_test = X_test.drop('AnnualSalary', axis=1)
    
    # =========================
    # ESCALONAMENTO
    # =========================
    
    scaler = StandardScaler()
    
    cols_numericas = ['Age', 'AnnualSalary_log']
    
    X_train[cols_numericas] = scaler.fit_transform(X_train[cols_numericas])
    X_test[cols_numericas] = scaler.transform(X_test[cols_numericas])
    
    return X_train, X_test