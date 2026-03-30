from sklearn.model_selection import train_test_split

def split_data(df, target):
    
    X = df.drop(target, axis=1)
    y = df[target]
    
    return train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )