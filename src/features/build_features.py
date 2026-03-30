def build_features(df):
    
    df = df.copy()  # segurança
    
    # remover coluna irrelevante
    df = df.drop(columns=['User ID'])
    
    # transformar variável categórica
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    
    return df