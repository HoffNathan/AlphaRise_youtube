import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

'''''
Links: 
https://www.gov.br/cvm/pt-br/assuntos/noticias/2021/cvm-reforca-alerta-de-atuacao-irregular-da-iq-option-ltd
https://www.gov.br/cvm/pt-br/assuntos/noticias/2022/cvm-reitera-alerta-relativo-a-atuacao-irregular-no-mercado-de-valores-mobiliarios

'''''

# Estilo dos gráficos
sns.set(style="darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)

# 1. BAIXA DADOS REAIS DO MERCADO
ticker = "^BVSP"
df = yf.download(ticker, period="10y", interval="1d")


df['return'] = df['Close'].pct_change()

df['return'].describe()

df['return'].plot()

payout = 1.00  # ganho quando acerta
loss = -1.00   # perda quando erra
prob_win = 0.5 # probabilidade de ganhar
prob_loss = (1 - prob_win) # probabilidade de perder
n_rodadas = 1000      # número de apostas simuladas
aposta = 5          # valor apostado em cada rodada
banca_inicial = 100 # valor inicial da banca

esperanca = (prob_win * payout) + (prob_loss * loss)

print(f"🎲 Esperança matemática por operação: ({esperanca*100:.1f}%)")

# Simula os resultados das apostas (payout ou perda)
resultados = np.random.choice([payout, loss], size=n_rodadas, p=[prob_win, prob_loss])

# Inicializa a banca e histórico de valores
banca = banca_inicial
banca_serie = [banca]

# Atualiza a banca rodada a rodada
for resultado in resultados:
    banca += resultado * aposta
    if banca <= 0:
        break  # Quebrou a banca, para a simulação
    banca_serie.append(banca)

# Cria DataFrame com histórico da banca
df_banca = pd.DataFrame({
    'Rodada': range(len(banca_serie)),
    'Banca': banca_serie
})

# Função para definir cor dos pontos no gráfico
def cor_banca(valor):
    if valor >= banca_inicial:
        return 'green'
    elif valor > banca_inicial * 0.3:
        return 'yellow'
    else:
        return 'red'

# Aplica a função para obter cores para cada ponto
colors = df_banca['Banca'].apply(cor_banca)

# Cria o gráfico interativo com Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_banca['Rodada'],
    y=df_banca['Banca'],
    mode='lines+markers',
    name='Banca',
    line=dict(color='blue', width=3),
    marker=dict(color=colors, size=8)
))

# Linha horizontal indicando banca inicial
fig.add_shape(type="line",
              x0=0, y0=banca_inicial,
              x1=len(banca_serie), y1=banca_inicial,
              line=dict(color="black", width=2, dash="dash"),
              name="Banca Inicial")

# Configuração do layout
fig.update_layout(
    title="Simulação de Apostas Binárias - Evolução da Banca",
    xaxis_title="Rodadas",
    yaxis_title="Valor da Banca (R$)",
    width=900,
    height=600,
    hovermode="x unified",
    template="plotly_white"
)

fig.show()

max_period = 1000  # limite máximo para o eixo x

def simula_binarias_trajetoria(payout=payout, loss=loss, prob_win=prob_win,
                               n_rodadas=n_rodadas, aposta=aposta, banca_inicial=banca_inicial):
    banca = banca_inicial
    historico = [banca]
    for _ in range(n_rodadas):
        resultado = np.random.choice([payout, loss], p=[prob_win, prob_loss])
        banca += resultado * aposta
        if banca <= 0:
            historico.append(0)
            break  # Quebrou
        historico.append(banca)
    while len(historico) < n_rodadas + 1:
        historico.append(0)
    return historico

n_simulacoes = 500
trajetorias = [simula_binarias_trajetoria() for _ in range(n_simulacoes)]
df_traj = pd.DataFrame(trajetorias).T

# Análise dos resultados finais
resultados_finais = df_traj.iloc[-1]
percentual_quebraram = (resultados_finais == 0).mean()
media_final = resultados_finais[resultados_finais > 0].mean() if (resultados_finais > 0).any() else 0

print(f"💀 % de pessoas que quebraram: {percentual_quebraram:.2%}")
print(f"📉 Banca média final (apenas quem não quebrou): R$ {media_final:.2f}")

# Ajustar para nunca ficar abaixo de 0 no gráfico (stop visual)
df_traj_clip = df_traj.clip(lower=0)

fig = go.Figure()

for col in df_traj_clip.columns:
    fig.add_trace(go.Scatter(
        x=df_traj_clip.index,
        y=df_traj_clip[col],
        mode='lines',
        line=dict(width=1),
        opacity=0.6,
        showlegend=False
    ))

fig.add_shape(type="line",
              x0=0, y0=banca_inicial,
              x1=max_period, y1=banca_inicial,
              line=dict(color="black", width=2, dash="dash"),
              name="Banca Inicial")

fig.update_layout(
    title="Simulação Monte Carlo - Evolução da Banca (stop em 0)",
    xaxis_title="Rodadas",
    yaxis_title="Valor da Banca (R$)",
    width=900,
    height=600,
    hovermode="x unified",
    template="plotly_white",
    xaxis=dict(range=[0, max_period]),
    yaxis=dict(range=[0, banca_inicial*1.2])  # limite visual do eixo y
)

fig.show()


# Exemplo: sua série de retornos periódicos (pode ser coluna do seu df)
returns = df['return']  # ajustar aqui para sua coluna real

# Calcula o valor acumulado da banca com juros compostos
valor_acumulado = banca_inicial * (1 + returns).cumprod()

# Monta o DataFrame para plotagem
df_valor = pd.DataFrame({
    'Rodada': returns.index,
    'Valor Acumulado': valor_acumulado
})

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_valor['Rodada'],
    y=df_valor['Valor Acumulado'],
    mode='lines',  # só linhas, sem markers
    name='Banca com Juros Compostos',
    line=dict(color='blue', width=3)
))

# Linha da banca inicial
fig.add_shape(type="line",
              x0=df_valor['Rodada'].min(),
              y0=banca_inicial,
              x1=df_valor['Rodada'].max(),
              y1=banca_inicial,
              line=dict(color="black", width=2, dash="dash"),
              name="Banca Inicial")

fig.update_layout(
    title="Evolução da Banca com Juros Compostos",
    xaxis_title="Rodadas",
    yaxis_title="Valor da Banca (R$)",
    width=900,
    height=600,
    hovermode="x unified",
    template="plotly_white"
)

fig.show()


# Jogo de soma 0 (não dá liquidez, não arbitra diferenças, não auxilia na constituição de dívida)
# Joga contra a corretora
# Não é regulamentada (de onde vem a cotação? kkkkk)
