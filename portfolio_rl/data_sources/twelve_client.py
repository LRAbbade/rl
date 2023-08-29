import os
from datetime import date
from typing import List
import requests
import pandas as pd


def add_ticker(ticker, to):
    to['ticker'] = ticker
    return to


class TwelveClient:
    base_url = 'https://api.twelvedata.com'

    def __init__(self, api_key: str, cache_folder: str):
        self.headers = {'Authorization': f'apikey {api_key}'}
        self.cache_folder = cache_folder

    def _dt_to_str(self, dt: date) -> str:
        return str(dt).replace('-', '')

    def cache_file(self, ticker: str, start: date, end: date, interval: str) -> str:
        return f'{self.cache_folder}/twelve_{ticker}_{self._dt_to_str(start)}_{self._dt_to_str(end)}_{interval}.parquet'

    def parse_multiple_tickers(self, res_json) -> pd.DataFrame:
        return pd.DataFrame([
            j for i in
                ((add_ticker(key, row) for row in data['values'])
                for key, data in res_json.items())
            for j in i
        ])

    def time_series(self, tickers: List[str], start: date, end: date, interval: str = '1day') -> pd.DataFrame:
        r = requests.get(self.base_url + '/time_series', {
            'symbol': ','.join(tickers),
            'interval': interval,
            'start_date': str(start),
            'end_date': str(end),
            'order': 'ASC'
        }, headers=self.headers)
        res_json = r.json()
        assert r.status_code == 200, r.text
        if 'code' in res_json:
            assert res_json['code'] == 200, res_json['message']

        if len(tickers) > 1:
            df = self.parse_multiple_tickers(res_json)
        else:
            df = add_ticker(res_json['meta']['symbol'], pd.DataFrame(res_json['values']))

        df['datetime'] = df['datetime'].map(pd.Timestamp)
        df['date'] = df['datetime'].map(lambda ts: ts.date())
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df

    def cached_single_time_series(self, ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
        file = self.cache_file(ticker, start, end, interval)
        if os.path.isfile(file):
            print(f'Found cached file {file}')
            return pd.read_parquet(file)
        else:
            df = self.time_series([ticker], start, end, interval)
            print(f'Saving data result in file {file}')
            df.to_parquet(file)
            return df
