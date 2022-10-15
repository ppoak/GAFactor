# %%
import pandas as pd

# %%
start = "2018-01-01"
end = "2022-01-01"
data = pd.read_parquet('data/kline_daily.parquet')

# %%
adj_open = data.loc[start:end, "open"] * data.loc[start:end, "back_adjfactor"]
adj_open = adj_open.unstack()
adj_open.to_parquet('data/raw_factor/adj_open.parquet')

# %%
adj_close = data.loc[start:end, "close"] * data.loc[start:end, "back_adjfactor"]
adj_close = adj_close.unstack()
adj_close.to_parquet('data/raw_factor/adj_close.parquet')

# %%
adj_high = data.loc[start:end, "high"] * data.loc[start:end, "back_adjfactor"]
adj_high = adj_high.unstack()
adj_high.to_parquet('data/raw_factor/adj_high.parquet')

# %%
adj_low = data.loc[start:end, "low"] * data.loc[start:end, "back_adjfactor"]
adj_low = adj_low.unstack()
adj_low.to_parquet('data/raw_factor/adj_low.parquet')

# %%
volume = data.loc[start:end, "volume"].unstack()
volume.to_parquet('data/raw_factor/volume.parquet')

# %%
label = data.groupby(level=1)['open'].shift(-2) / data.groupby(level=1)['open'].shift(-1) - 1
label = label.loc[start:end].unstack()
label.to_parquet('data/raw_factor/label.parquet')
