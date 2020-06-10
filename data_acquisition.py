from datetime import datetime, timedelta, timezone
import pandas as pd
from tenacity import retry, wait_fixed, stop_after_delay
import requests
import json
from google.cloud import bigquery
import numpy as np

def generate_date_intervals(start_date, end_date, block_size, block_period_s):
    from datetime import timedelta
    """
    Generate a series of date intervals of `block_size` elements each of length `block_period_s`.
    Used for generating contiguous start,end date periods for api queries that exceed limit parameter.
    `start_date` and `end_date` are both expected as `datetime` objects.
    """
    td = (end_date-start_date)
    n_periods = 0

    if hasattr(td, 'days'):
        n_periods += td.days*(24*60*60/block_period_s)
    if hasattr(td, 'seconds'):
        n_periods += td.seconds/block_period_s

    date_ranges=[]

    assert end_date > start_date, "End date and start date are in the incorrect order"
    for i in np.arange(np.ceil(n_periods/block_size)):
        st_i = start_date+timedelta(seconds=(block_size)*i*block_period_s) # start time i
        et_i = min(st_i+timedelta(seconds=block_size*block_period_s-1), end_date) # end time i
        date_ranges.append([st_i, et_i])

    return date_ranges, n_periods

def get_ohlcv_binance_futures(sym, start, end, period='5m'):
    url_base = 'https://fapi.binance.com/fapi/v1/klines'
    mts = lambda dt: int(dt.timestamp()*1000) #ms timestamp
    batch_size = 1400
    date_ranges, n_periods = generate_date_intervals(start,end,batch_size, 5*60)
    period = '5m'
    data = []

    for s, e in date_ranges:
        req = requests.get(f"{url_base}?symbol={sym}&interval={period}&startTime={mts(s)}&endTime={mts(e)}&limit={batch_size}")
        data += json.loads(req.text)

    df = pd.DataFrame(np.array(data)[:, :6], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df.timestamp = pd.to_datetime(df.timestamp, unit='ms')
    return df.set_index('timestamp').astype('float')

def get_binance_futures_feats(feat_info, start, end, limit=None, sym='BTCUSDT'):
    """
    Gets binance futures trading data such as open interest and sentiment data (long/short ratios etc).

    :param start: (datetime) start date - must be within last 30 days
    :param end: (datetime) end date - must be greater than start date
    :param feat_info: a list of feature dicts. Each dict corresponds to a feauture and should have the following form:
    ```
    {
        'name': open_int,
        'api_suffix': '/openInterestHist'
        'col_modifier': None
    }
    ```
    The above is an example. Clearly, these features must all be available on binance futures api. The col_modifier
    simply appends a custom string to all features for that endpoint. 

    ref: https://binance-docs.github.io/apidocs/futures/en/#top-trader-long-short-ratio-positions-market_data

    Note that Binance only allows fetching of data for last 30 days.
    """
    mts = lambda dt: int(dt.timestamp()*1000) #ms timestamp
    batch_size = 400
    dfs = []
    date_ranges, n_periods = generate_date_intervals(start,end,batch_size, 5*60)
    api_base = 'https://fapi.binance.com/futures/data'
    period = '5m'

    for start, end in date_ranges:
        df = None
        s = mts(start)
        e = mts(end)
        for feat in feat_info:
            lim = limit if limit is not None else batch_size
            url = f"{api_base}{feat['api_suffix']}?symbol={sym}&period={period}&startTime={s}&endTime={e}&limit={lim}"
            req = requests.get(url)
            d = json.loads(req.text)
            if req.status_code != 200 or not isinstance(d, list) or len(d)==0:
                logging.error(f"Error: request to {url} received unexpected response: {d}")
                return None
            dfi = pd.DataFrame(d).set_index('timestamp')
            if 'symbol' in dfi.columns:
                dfi.drop(columns=['symbol'], inplace=True)
            if feat['col_modifier'] is not None:
                dfi = dfi.rename(columns={c:c+feat['col_modifier'] for c in dfi.columns})
            if df is None:
                df = dfi
            else:
                df = df.join(dfi)

        df.reset_index(inplace=True)
        df.timestamp = pd.to_datetime(df.timestamp, unit='ms')
        dfs.append(df)

    if len(dfs)>0:
        print("Combining data across batches...")
        df = dfs[0]
        for _df in dfs[1:]:
            df = pd.concat([df, _df], ignore_index=True)
            print(_df.timestamp.iloc[0], _df.timestamp.iloc[-1])

    return df.set_index('timestamp').astype('float')

@retry(wait=wait_fixed(15), stop=stop_after_delay(20))
def fetch_data(start, feat_info, end=None, price_key='close', sym='BTCUSDT'):
    if end is None: 
        end = datetime.now() # set to current time if not supplied
    sent_data = get_binance_futures_feats(feat_info, start, end, sym=sym) # sentiment data
    ohlcv = get_ohlcv_binance_futures(sym, start, end)
    if sent_data is None or ohlcv is None:
        raise ValueError("Failed to retrieve data completely")
    elif sent_data.index[-1]!=ohlcv.index[-1]:
        raise ValueError("Sentiment and pricing data not synced")
    return sent_data.join(ohlcv[[price_key,'volume']])


def load_latest_futures_sent_data(filepath, feat_info=None, merge_data=True, save=False):
    """
    Function to continually load latest sentiment data from binance futures. Note, 
    this api only allows data as far back as last 30 days. 

    :param filepath: either path to a save a new file at or the path of an existing file to be appended to
    :param feat_ino: dict of feats of same form as the version in `get_binance_futures_feats`
    :param merge_data: if a filepath to a previous data file is given and this is true, new data will be merged to old and returned
    :param save: whether or not to save (create new or append) to file

    :return DataFrame of binance futures data
    """
    from os import path

    if feat_info is None:
        feat_info =[
            ['open_int', '/openInterestHist', None],
            ['top_lsr_acc', '/topLongShortAccountRatio', '_top_acc'],
            ['top_lsr_pos', '/topLongShortPositionRatio', '_top_pos'],
            ['lsr_acc', '/globalLongShortAccountRatio', '_global'],
            ['taker_vol', '/takerlongshortRatio', None]
          ]
        feat_info = [dict(zip(['name', 'api_suffix', 'col_modifier'], f)) for f in feat_info]

    end = datetime.now()
    old_data = None

    if path.exists(filepath) and filepath.endswith('.parquet'):
        old_data = pd.read_parquet(filepath)
        start = old_data.index[-1]
        print(f"Found existing file. Retrieving new data from {start} until now ({end})")
    else:
        start = end - timedelta(days=29, hours=22) # approx 30 days accounting for time zone weirdness on binance

    data = fetch_data(start, feat_info, end=end)

    if merge_data and old_data is not None:
        data = pd.concat([old_data, data[data.index>old_data.index[-1]]], sort=False)

    if save:
        print(f"Saving data to {filepath}")
        data.to_parquet(filepath)

    return data

def main():
    feat_info =[
        ['open_int', '/openInterestHist', None],
        ['top_lsr_acc', '/topLongShortAccountRatio', '_top_acc'],
        ['top_lsr_pos', '/topLongShortPositionRatio', '_top_pos'],
        ['lsr_acc', '/globalLongShortAccountRatio', '_global'],
        ['taker_vol', '/takerlongshortRatio', None]
      ]
    feat_info = [dict(zip(['name', 'api_suffix', 'col_modifier'], f)) for f in feat_info]

    # Define bigquery client
    client = bigquery.Client()
    table_id = 'sentiment_data.binance_sentiment_BTCUSDT'

    # Obtain last timestamp
    query = (
        "SELECT max(timestamp) as last_time FROM `invictus-dev.sentiment_data.binance_sentiment`"
    )
    query_job = client.query(query)
    last_time = list(query_job.result())[0].last_time

    end = datetime.now(timezone.utc)
    start = last_time + timedelta(minutes=5)
    df = fetch_data(start, feat_info, end=end, sym='BTCUSDT')

    print(last_time)
    print(df)
    print(len(df))
    print(df.dropna())
    print(len(df.dropna()))
    print(df[df.isna().any(axis=1)])

    # job = client.load_table_from_dataframe(
        # df, table_id
    # )


if __name__=='__main__':
    main()
