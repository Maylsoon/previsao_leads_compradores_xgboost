import pandas as pd
import matplotlib.pyplot as plt

def compare_models(scores_log, scores_xgb):
    
    df_result = pd.DataFrame({
        'Modelo': ['Regressão Logística', 'XGBoost'],
        'ROC AUC média': [scores_log.mean(), scores_xgb.mean()],
        'Desvio padrão': [scores_log.std(), scores_xgb.std()]
    }).sort_values(by='ROC AUC média', ascending=False)
    
    print(df_result.head())
    
    # gráfico
    plt.bar(df_result['Modelo'], df_result['ROC AUC média'])
    plt.ylabel('ROC AUC')
    plt.title('Comparação de modelos')
    plt.show()
    
    return df_result