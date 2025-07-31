#!pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git

# Imports
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from tvDatafeed import TvDatafeed, Interval
from datetime import date
get_ipython().run_line_magic("matplotlib", "inline")


import warnings
warnings.filterwarnings("ignore")

TICKER1 = "WIN1!"
EXCHANGE = "bmfbovespa"
N_BARS = 10000

# === COLETA DE DADOS COM TVDATAFEED ===
tv = TvDatafeed()

# Download da base
df1 = (
    tv.get_hist(symbol=TICKER1, exchange=EXCHANGE, interval=Interval.in_daily, n_bars=N_BARS)
    .drop(columns=['volume', 'symbol']) #jogando fora
)


# Ajuste nas colunas
df1.columns = [f"{TICKER1}_{col}" for col in df1.columns]

df1.index = pd.to_datetime(df1.index).normalize()




# Get the data
df1["Returns"] = df1[f"{TICKER1}_close"].pct_change(1)
df1["Adj Close"] = df1[f"{TICKER1}_close"]
df1["Adj Low"] = df1[f"{TICKER1}_low"]
df1["Adj High"] = df1[f"{TICKER1}_high"]
df1["Adj Open"] = df1[f"{TICKER1}_open"]

vol_p1 = 20

df1["Vol"] = np.round(df1["Returns"].rolling(vol_p1).std()*np.sqrt(252), 4)

# Annual S&D Volatility Zones

# For the Annualized Volatility Chart

fig = make_subplots(rows = 1, cols = 1,
                    shared_xaxes = True,
                    vertical_spacing = 0.08)

fig.add_trace(go.Scatter(x = df1.index, y = df1["Vol"]*100
                         , name = "Vol", line = dict(color = "blue"))
              , row = 1, col = 1)


fig.update_layout(height = 600, width = 800
                  , title_text = "Annualized Volatility - " + TICKER1
                  , font_color = "blue"
                  , title_font_color = "black"
                  , xaxis_title = "Years"
                  , yaxis_title = "Volatility (%)"
                  , legend_title = "Vol"
                  , font = dict(size = 15, color = "Black")
                 )
fig.update_layout(hovermode = "x")

# Code to exclude empty dates from the chart
dt_all = pd.date_range(start = df1.index[0]
                       , end = df1.index[-1]
                       , freq = "D")
dt_all_py = [d.to_pydatetime() for d in dt_all]
dt_obs_py = [d.to_pydatetime() for d in df1.index]

dt_breaks = [d for d in dt_all_py if d not in dt_obs_py]

fig.update_xaxes(
    rangebreaks = [dict(values = dt_breaks)]
)
fig.show()

import plotly.express as px

fig = px.histogram(df1, x="Vol", title="Histogram of Volatility")
fig.update_layout(xaxis_title="Volatility", yaxis_title="Frequency", height=800, width=800)
fig.show()

year = "2019"


df_year = df1[df1.index.year == int(year)]
df_next_year = df1[df1.index.year == int(year) + 1]


Upper_Band_12m1d =  df_year["Vol"][-1]*df_year["Adj Close"][-1] + df_year["Adj Close"][-1]
Lower_Band_12m1d =  df_year["Adj Close"][-1] - df_year["Vol"][-1]*df_year["Adj Close"][-1]

Upper_Band_12m2d =  2*df_year["Vol"][-1]*df_year["Adj Close"][-1] + df_year["Adj Close"][-1]
Lower_Band_12m2d =  df_year["Adj Close"][-1] - 2*df_year["Vol"][-1]*df_year["Adj Close"][-1]

Upper_Band_12m3d =  3*df_year["Vol"][-1]*df_year["Adj Close"][-1] + df_year["Adj Close"][-1]
Lower_Band_12m4d =  df_year["Adj Close"][-1] - 4*df_year["Vol"][-1]*df_year["Adj Close"][-1]


# Annual S&D Volatility Zones chart

fig = make_subplots(rows = 1, cols = 1,
                    shared_xaxes = True,
                    vertical_spacing = 0.08)


fig.add_trace(go.Candlestick(x=df_next_year.index,
  open=df_next_year["Adj Open"], high=df_next_year["Adj High"],
  low=df_next_year["Adj Low"], close=df_next_year["Adj Close"],
  name="Candle",
  increasing_line_color="black", decreasing_line_color="red"),
  row=1, col=1)

fig.add_hline(y = Upper_Band_12m1d, line_width = 1, line_dash = "dash", line_color = "green")
fig.add_hline(y = Lower_Band_12m1d, line_width = 1, line_dash = "dash", line_color = "red")

fig.add_hline(y = Upper_Band_12m2d, line_width = 1, line_dash = "dash", line_color = "green")
fig.add_hline(y = Lower_Band_12m2d, line_width = 1, line_dash = "dash", line_color = "red")

fig.add_hline(y = Upper_Band_12m3d, line_width = 2, line_dash = "dash", line_color = "green")
fig.add_hline(y = Lower_Band_12m4d, line_width = 2, line_dash = "dash", line_color = "red")

fig.update_layout(height = 600, width = 800
                  , title_text = "Annual S&D Volatility Zones: " + str(int(year)+1) + " " + TICKER1
                  , font_color = "blue"
                  , title_font_color = "black"
                  , xaxis_title = "Years"
                  , yaxis_title = "Close"
                  , legend_title = "Vol"
                  , font = dict(size = 15, color = "Black")
                  , xaxis_rangeslider_visible = False
                 )
fig.update_layout(hovermode = "x")

# Code to exclude empty dates from the chart
dt_all = pd.date_range(start = df1.index[0]
                       , end = df1.index[-1]
                       , freq = "D")
dt_all_py = [d.to_pydatetime() for d in dt_all]
dt_obs_py = [d.to_pydatetime() for d in df1.index]

dt_breaks = [d for d in dt_all_py if d not in dt_obs_py]

fig.update_xaxes(
    rangebreaks = [dict(values = dt_breaks)]
)


fig.show()
