#!pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git

# Imports
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from tvDatafeed import TvDatafeed, Interval
from numpy import mean, absolute
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings("ignore")

TICKER1 = "abev3"
EXCHANGE = "bmfbovespa"
N_BARS = 10000
VOL_PRED = 10

# === COLETA DE DADOS COM TVDATAFEED ===
tv = TvDatafeed()

# Download da base
df1 = (
    tv.get_hist(symbol=TICKER1, exchange=EXCHANGE, interval=Interval.in_daily, n_bars=N_BARS)
    .drop(columns=['volume', 'symbol']) #jogando fora
)


# target
df1["target(%)"] = df1["close"].shift(-VOL_PRED) - df1["close"]

# target vol
df1['Returns'] = df1["close"].pct_change()

df1['HistVol'] = df1['Returns'].rolling(window=VOL_PRED).std()*100

df1['FutureVolatility'] = df1['HistVol'].shift(-VOL_PRED)


#vars

# Médias móveis
df1['MA5'] = df1['close'].rolling(5).mean()
df1['MA10'] = df1['close'].rolling(10).mean()
df1['MA15'] = df1['close'].rolling(15).mean()
df1['MA20'] = df1['close'].rolling(20).mean()

# RSL (Close / MA) - 1
df1['RSL_5'] = (df1['close'] / df1['MA5']) - 1
df1['RSL_10'] = (df1['close'] / df1['MA10']) - 1
df1['RSL_15'] = (df1['close'] / df1['MA15']) - 1

# var e STD
df1["Return_var5"] = df1["Returns"].rolling(5).var()
df1["Return_var10"] = df1["Returns"].rolling(10).var()

df1["Return_std5"] = df1["Returns"].rolling(5).std()
df1["Return_std10"] = df1["Returns"].rolling(10).std()



#base manipulation

start_train = "2006-01-01"
end_train = "2018-12-31"

start_test = "2018-12-31"
end_test = "2025-12-31"

df_filtrado = df1.iloc[VOL_PRED*20:]

df1_train = df_filtrado.loc[start_train : end_train]


df1_test = df_filtrado.loc[start_test : end_test]


features = [

    # RSL (Close / MA) - 1
    'RSL_5', 'RSL_10', 'RSL_15',

    # var e STD
    'Return_var5', 'Return_var10',
    'Return_std5', 'Return_std10',

]


target = "FutureVolatility"

X_train = df1_train[features]
y_train = df1_train[target]

X_test = df1_test[features]
y_test = df1_test[target]

# 8. Treinar o modelo Gradient Boosting
model = GradientBoostingRegressor(
    n_estimators=1500,        # Mais árvores
    learning_rate=0.05,      # Aprendizado mais suave
    max_depth=3,             # Árvores mais ou menos profundas
    min_samples_leaf=5,      # Ajuda contra o overfitting
    min_samples_split=10,    # Garante divisões relevantes
    subsample=0.8,           # Usa 80% dos dados para cada árvore
    max_features='sqrt',     # Usa a raiz quadrada do nº de features
    random_state=42
)

model.fit(X_train, y_train)

# 9. Fazer previsões para o período de teste
df_filtrado['Predicted_FutureVolatility'] = model.predict(df_filtrado[features])
df1_train['Predicted_FutureVolatility'] = model.predict(df1_train[features])
df1_test['Predicted_FutureVolatility'] = model.predict(df1_test[features])

# 10. Plotar os resultados (volatilidade futura observada vs prevista) usando Plotly

first_test_date = df1_test.index.min()  # Obtém a primeira data do subset test

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)

fig.add_trace(go.Scatter(x=df_filtrado.index, y=df_filtrado["HistVol"].rolling(VOL_PRED).mean(),
                         mode='lines',
                         name='Volatilidade Observada'), row=1, col=1)
fig.add_trace(go.Scatter(x=df_filtrado.index, y=df_filtrado['Predicted_FutureVolatility'].rolling(VOL_PRED).mean(), ###linha que desloca
                         mode='lines',
                         name='Volatilidade Futura Prevista'), row=1, col=1)

fig.add_trace(go.Scatter(x=df_filtrado.index, y=df_filtrado['close'].values,
                         mode='lines',
                         name='Close'), row=2, col=1)

fig.add_shape(
    dict(
        type="line",
        x0=first_test_date, x1=first_test_date,  # A linha vertical na mesma posição x
        y0=df_filtrado["HistVol"].min(), y1=df_filtrado["HistVol"].max(),  # Altura da linha cobrindo todo o eixo Y
        line=dict(color="red", width=2, dash="dash")  # Linha vermelha tracejada
    ), row=1, col=1
)

fig.update_layout(title= "Vol observada",
                  xaxis2_title='Data',
                  yaxis_title='Volatilidade',
                  yaxis2_title='Close',
                  template='plotly_white',
                  font=dict(size=15, color="Black"),
                  height=900, width=1200)

fig.show()

#verificando acurácia
df1_test_acc = df1_test.dropna()

df1_test_acc['Binary_Target_observado'] = np.where(df1_test_acc['FutureVolatility'] > df1_test_acc['HistVol'],1,0)
df1_test_acc['Binary_Target_previsto'] = np.where(df1_test_acc['Predicted_FutureVolatility'] > df1_test_acc['HistVol'],1,0)

y_true = df1_test_acc['Binary_Target_observado']
y_pred = df1_test_acc['Binary_Target_previsto']

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# Métricas
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall_1 = recall_score(y_true, y_pred, pos_label=1)
recall_0 = recall_score(y_true, y_pred, pos_label=0)
f1 = f1_score(y_true, y_pred)

# Impressão estruturada
print("Confusion Matrix:")
print(f"[[TN: {tn}  FP: {fp}]\n [FN: {fn}  TP: {tp}]]")

print("\nDesempenho Geral:")
print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision (1) : {precision:.4f}")
print(f"Recall    (1) : {recall_1:.4f}")
print(f"Recall    (0) : {recall_0:.4f}")
print(f"F1-Score      : {f1:.4f}")

print(f"precision = TP / (TP + FP) = {tn} / {tn+fn} ≈ {(tn / (tn+fn)):.1%}")

print(f"precision = TP / (TP + FP) = {tp} / {tp+fp} ≈ {(tp / (tp + fp)):.1%}")

# Get feature importances from the GradientBoostingRegressor model
importance = model.feature_importances_

# Assuming 'features' is the list of feature names used for training
feature_names = features

# Create a pandas Series for easier sorting and selection
feature_importance_series = pd.Series(importance, index=feature_names)

# Sort feature importances and select the top 25
top_n = 25
top_important_vars = feature_importance_series.sort_values(ascending=False).head(top_n)

# Plot the feature importances
plt.figure(figsize=(10, 8))
top_important_vars.plot(kind='barh')
plt.title('Top 25 Feature Importances (GradientBoostingRegressor)')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.gca().invert_yaxis() # Invert y-axis to show the most important feature at the top
plt.show()
