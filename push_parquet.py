from google.cloud import bigquery as bq
import pandas as pd

client = bq.Client()

symbols = ["btc", "eth", "bch", "xrp"]
for s in symbols:
    df = pd.read_parquet(f"data/bin_futures_{s}.parquet")
    table_id = f"sentiment_data.binance_sentiment_{s.upper()}USDT"
    client.load_table_from_dataframe(df, table_id)
