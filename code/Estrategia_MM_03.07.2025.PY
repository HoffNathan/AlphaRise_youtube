#!pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git
#!pip install catboost

from tvDatafeed import TvDatafeed, Interval
import numpy as np
import plotly.graph_objects as go
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# === PARÂMETROS GLOBAIS ===
TICKER1 = "WIN1!"
EXCHANGE = "BMFBOVESPA"
N_BARS = 10000
ALVO = 1

# === COLETA DE DADOS COM TVDATAFEED ===
tv = TvDatafeed()

# Download da base
df = (
    tv.get_hist(symbol=TICKER1, exchange=EXCHANGE, interval=Interval.in_daily, n_bars=N_BARS)
    .drop(columns=['volume', 'symbol']) #jogando fora
)


# Ajuste nas colunas
df.columns = [f"{TICKER1}_{col}" for col in df.columns]

df["Return"] = df[f"{TICKER1}_close"].pct_change(1)
df["Target"] = df["Return"].shift(-ALVO)

df["MM_05"] = df[f"{TICKER1}_close"].rolling(5).mean()
df["MM_10"] = df[f"{TICKER1}_close"].rolling(10).mean()
df["MM_20"] = df[f"{TICKER1}_close"].rolling(20).mean()
df["MM_52"] = df[f"{TICKER1}_close"].rolling(52).mean()
df["MM_200"] = df[f"{TICKER1}_close"].rolling(200).mean()

df["MM_05_20"] = np.where(df["MM_05"] > df["MM_20"],1,0)
df["MM_10_20"] = np.where(df["MM_10"] > df["MM_20"],1,0)
df["MM_20_52"] = np.where(df["MM_20"] > df["MM_52"],1,0)
df["MM_52_200"] = np.where(df["MM_52"] > df["MM_200"],1,0)
df["MM_20_200"] = np.where(df["MM_20"] > df["MM_200"],1,0)

#== backtest ===

df["MM_cruzamento"] = df["MM_05_20"].astype(str)
#df["MM_cruzamento"] = df["MM_10_20"].astype(str)
#df["MM_cruzamento"] = df["MM_20_52"].astype(str)
#df["MM_cruzamento"] = df["MM_52_200"].astype(str)
#df["MM_cruzamento"] = df["MM_20_200"].astype(str)


df["Retorno_short"] = np.where((df["MM_cruzamento"] == '0'), -df["Target"], 0)
df["Retorno_long"] = np.where((df["MM_cruzamento"] == '1'), df["Target"], 0)
df["Retorno_both"] = np.where((df["MM_cruzamento"] == '1'), df["Target"], df["Retorno_short"])

df["Retorno_short"] = df["Retorno_short"].astype(float)
df["Retorno_long"] = df["Retorno_long"].astype(float)
df["Retorno_both"] = df["Retorno_both"].astype(float)

df["Retorno_both_acm"] = df["Retorno_both"].cumsum()
df["Retorno_short_acm"] = df["Retorno_short"].cumsum()
df["Retorno_long_acm"] = df["Retorno_long"].cumsum()

# Calcula o retorno acumulado para a estratégia buy and hold
df["Retorno_buy_hold_acm"] = (df[f"{TICKER1}_close"] / df[f"{TICKER1}_close"].iloc[0])-1


# Criação do gráfico interativo
fig = go.Figure()

# Adiciona as linhas de retorno acumulado (both, short, long)
fig.add_trace(go.Scatter(x=df.index, y=df['Retorno_both_acm'], mode='lines', name='Retorno Acumulado (both)'))
fig.add_trace(go.Scatter(x=df.index, y=df['Retorno_short_acm'], mode='lines', name='Retorno Acumulado (short)'))
fig.add_trace(go.Scatter(x=df.index, y=df['Retorno_long_acm'], mode='lines', name='Retorno Acumulado (long)'))
fig.add_trace(go.Scatter(x=df.index, y=df["Retorno_buy_hold_acm"], mode='lines', name='Retorno Acumulado (buy n hold)'))


# Títulos e rótulos
fig.update_layout(
    title='Estratégia - WIN',
    xaxis_title='Data',
    yaxis_title='Retorno Acumulado',
    template="plotly_dark",  # Opcional: altere o tema
    xaxis_rangeslider_visible=False,  # Remove o range slider
    plot_bgcolor='rgba(0, 0, 0, 0)',  # Altera o fundo do gráfico
    width=900,  # Ajusta a largura do gráfico
    height=500,  # Ajusta a altura do gráfico
)

# Exibe o gráfico
fig.show()

# === COLETA DE DADOS COM TVDATAFEED ===
tv = TvDatafeed()

# Download da base
df1 = (
    tv.get_hist(symbol=TICKER1, exchange=EXCHANGE, interval=Interval.in_5_minute, n_bars=N_BARS)
    .drop(columns=['volume', 'symbol']) #jogando fora
)

# Ajuste nas colunas
df1.columns = [f"{TICKER1}_{col}" for col in df1.columns]

df1["Return"] = df1[f"{TICKER1}_close"].pct_change(1)
df1["Target"] = df1["Return"].shift(-ALVO)

df1["MM_05"] = df1[f"{TICKER1}_close"].rolling(5).mean()
df1["MM_10"] = df1[f"{TICKER1}_close"].rolling(10).mean()
df1["MM_20"] = df1[f"{TICKER1}_close"].rolling(20).mean()
df1["MM_52"] = df1[f"{TICKER1}_close"].rolling(52).mean()
df1["MM_200"] = df1[f"{TICKER1}_close"].rolling(200).mean()

df1["MM_05_20"] = np.where(df1["MM_05"] > df1["MM_10"],1,0)
df1["MM_10_20"] = np.where(df1["MM_10"] > df1["MM_20"],1,0)
df1["MM_20_52"] = np.where(df1["MM_20"] > df1["MM_52"],1,0)
df1["MM_52_200"] = np.where(df1["MM_52"] > df1["MM_200"],1,0)
df1["MM_20_200"] = np.where(df1["MM_20"] > df1["MM_200"],1,0)

#== backtest ===

df1["MM_cruzamento"] = df1["MM_05_20"].astype(str)
#df1["MM_cruzamento"] = df1["MM_10_20"].astype(str)
#df1["MM_cruzamento"] = df1["MM_20_52"].astype(str)
#df1["MM_cruzamento"] = df1["MM_52_200"].astype(str)
#df1["MM_cruzamento"] = df1["MM_20_200"].astype(str)


df1["Retorno_short"] = np.where((df1["MM_cruzamento"] == '0'), -df1["Target"], 0)
df1["Retorno_long"] = np.where((df1["MM_cruzamento"] == '1'), df1["Target"], 0)
df1["Retorno_both"] = np.where((df1["MM_cruzamento"] == '1'), df1["Target"], df1["Retorno_short"])

df1["Retorno_short"] = df1["Retorno_short"].astype(float)
df1["Retorno_long"] = df1["Retorno_long"].astype(float)
df1["Retorno_both"] = df1["Retorno_both"].astype(float)

df1["Retorno_both_acm"] = df1["Retorno_both"].cumsum()
df1["Retorno_short_acm"] = df1["Retorno_short"].cumsum()
df1["Retorno_long_acm"] = df1["Retorno_long"].cumsum()

# Calcula o retorno acumulado para a estratégia buy and hold
df1["Retorno_buy_hold_acm"] = (df1[f"{TICKER1}_close"] / df1[f"{TICKER1}_close"].iloc[0])-1


# Criação do gráfico interativo
fig = go.Figure()

# Adiciona as linhas de retorno acumulado (both, short, long)
fig.add_trace(go.Scatter(x=df1.index, y=df1['Retorno_both_acm'], mode='lines', name='Retorno Acumulado (both)'))
fig.add_trace(go.Scatter(x=df1.index, y=df1['Retorno_short_acm'], mode='lines', name='Retorno Acumulado (short)'))
fig.add_trace(go.Scatter(x=df1.index, y=df1['Retorno_long_acm'], mode='lines', name='Retorno Acumulado (long)'))
fig.add_trace(go.Scatter(x=df1.index, y=df1["Retorno_buy_hold_acm"], mode='lines', name='Retorno Acumulado (buy n hold)'))


# Títulos e rótulos
fig.update_layout(
    title='Estratégia - WIN',
    xaxis_title='Data',
    yaxis_title='Retorno Acumulado',
    template="plotly_dark",  # Opcional: altere o tema
    xaxis_rangeslider_visible=False,  # Remove o range slider
    plot_bgcolor='rgba(0, 0, 0, 0)',  # Altera o fundo do gráfico
    width=900,  # Ajusta a largura do gráfico
    height=500,  # Ajusta a altura do gráfico
)

# Exibe o gráfico
fig.show()

#features modelo

df["MM_10_20_model"] = df["MM_10"] / df["MM_20"]
df["MM_52_200_model"] = df["MM_52"] / df["MM_200"]


df["Return_std"] = df["Return"].rolling(10).std()
df["Max_min"] = df[f"{TICKER1}_high"] / df[f"{TICKER1}_low"]


df["Target_bi"] = np.where(df["Target"] > 0, 1, 0)

features = ["MM_52_200_model", "MM_10_20_model", "Return_std","Max_min" ]

# Drop rows with NaN in features or target
df_cleaned = df.dropna(subset=features + ["Target"])

X = df_cleaned[features]
y = df_cleaned["Target"]

# Separando os dados entre treinamento e teste

start_train = "2006-01-01"
end_train = "2013-12-31"

start_test = "2013-12-31"
end_test = "2025-12-31"

df_train = df_cleaned.loc[start_train : end_train]


df_test = df_cleaned.loc[start_test : end_test]

x_train = df_train[features]
y_train = df_train["Target_bi"]

x_test = df_test[features]
y_test = df_test["Target_bi"]

# Inicializa o modelo CatBoostClassifier
model = CatBoostClassifier(iterations=100,  # Número de árvores (boostings)
                           learning_rate=0.1, # Taxa de aprendizado
                           depth=6,           # Profundidade das árvores
                           loss_function='Logloss', # Função de perda para classificação binária
                           eval_metric='Accuracy',  # Métrica de avaliação
                           random_seed=42,      # Semente para reproducibilidade
                           verbose=0)           # Define para 0 para não mostrar o progresso do treinamento

# Treina o modelo
model.fit(x_train, y_train)
modelo_previsao_train = model.predict(x_train)
modelo_previsao = model.predict(x_test)


# Faz previsões no conjunto de teste
y_pred = model.predict(x_test)

# Calcula e printa a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"A acurácia do modelo no conjunto de teste é: {accuracy}")

df_train["modelo_previsao_train"] = modelo_previsao_train
pv = pd.pivot_table(
   df_train.loc[df_train["modelo_previsao_train"].notnull()],
   index=["modelo_previsao_train"],
   values=[f"Target"],
   aggfunc=["mean", "sum", "count"]
)

pv

#== backtest ===

df_test["model"] = modelo_previsao
df_test["model"] = df_test["model"].astype(str)

df_test["Retorno_short"] = np.where((df_test["model"] == '0'), -df_test[f"Target"], 0)
df_test["Retorno_long"] = np.where((df_test["model"] == '1'), df_test[f"Target"], 0)
df_test["Retorno_both"] = np.where((df_test["model"] == '1'), df_test[f"Target"], df_test["Retorno_short"])

df_test["Retorno_short"] = df_test["Retorno_short"].astype(float)
df_test["Retorno_long"] = df_test["Retorno_long"].astype(float)
df_test["Retorno_both"] = df_test["Retorno_both"].astype(float)

df_test["Retorno_both_acm"] = df_test["Retorno_both"].cumsum()
df_test["Retorno_short_acm"] = df_test["Retorno_short"].cumsum()
df_test["Retorno_long_acm"] = df_test["Retorno_long"].cumsum()

# Calcula o retorno acumulado para a estratégia buy and hold
df_test["Retorno_buy_hold_acm"] = (df_test[f"{TICKER1}_close"] / df_test[f"{TICKER1}_close"].iloc[0])-1


# Criação do gráfico interativo
fig = go.Figure()

# Adiciona as linhas de retorno acumulado (both, short, long)
fig.add_trace(go.Scatter(x=df_test.index, y=df_test['Retorno_both_acm'], mode='lines', name='Retorno Acumulado (both)'))
fig.add_trace(go.Scatter(x=df_test.index, y=df_test['Retorno_short_acm'], mode='lines', name='Retorno Acumulado (short)'))
fig.add_trace(go.Scatter(x=df_test.index, y=df_test['Retorno_long_acm'], mode='lines', name='Retorno Acumulado (long)'))
fig.add_trace(go.Scatter(x=df_test.index, y=df_test["Retorno_buy_hold_acm"], mode='lines', name='Retorno Acumulado (buy n hold)'))


# Títulos e rótulos
fig.update_layout(
    title='Estratégia - WIN',
    xaxis_title='Data',
    yaxis_title='Retorno Acumulado',
    template="plotly_dark",  # Opcional: altere o tema
    xaxis_rangeslider_visible=False,  # Remove o range slider
    plot_bgcolor='rgba(0, 0, 0, 0)',  # Altera o fundo do gráfico
    width=900,  # Ajusta a largura do gráfico
    height=500,  # Ajusta a altura do gráfico
)

# Exibe o gráfico
fig.show()
