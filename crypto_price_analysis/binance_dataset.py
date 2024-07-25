import os
from itertools import islice
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import json
import requests
from datetime import date, datetime, timedelta, timezone
import calendar
import hashlib

from .checksums import verify_file_checksum, create_file_checksum

import pandas as pd
import numpy as np



def incr_month(d: date) -> date:
	year = d.year + (d.month == 12)
	month = (d.month % 12) + 1
	day = d.day if d.day <= calendar.monthrange(year, month)[1] else 1
	return date(year, month, day)



class BinanceDataset:

	def __init__(self, directory: Path, ticker: str, interval: str, max_gap_size: int):
		self.directory = directory
		self.ticker = ticker
		self.interval = interval
		self.max_gap_size = max_gap_size
		self.ticker_interval_id = ticker + '-' + interval
		self.ticker_interval_directory = directory / self.ticker_interval_id

		match ticker:
			case 'ADAUSDT':
				self.default_date = date(2018, 4, 1)
			case 'BTCUSDT':   
				self.default_date = date(2017, 8, 1)
			case 'ETHUSDT':   
				self.default_date = date(2017, 8, 1)
			case 'SOLUSDT':   
				self.default_date = date(2020, 8, 1)
			case 'XRPUSDT':   
				self.default_date = date(2018, 8, 1)
			case _:
				raise ValueError('Unknown ticker')


	def download_csv_file(self, date_str: str, monthly: bool) -> bool:
			"""
			returns wether the data was downloaded successfully
			"""		
			base_file_name = f"{self.ticker + '-' + self.interval}-{date_str}"

			period = 'monthly' if monthly else 'daily'

			csv_file_url = f'https://data.binance.vision/data/spot/{period}/klines/{self.ticker}/{self.interval}/{base_file_name}.zip'
			checksum_file_url = csv_file_url + '.CHECKSUM'

			zip_data = requests.get(csv_file_url)
			provided_checksum = requests.get(checksum_file_url).text[:64]

			zip_data_checksum = hashlib.sha256(zip_data.content).hexdigest()

			if provided_checksum != zip_data_checksum:
				print('Failed to dowload', csv_file_url)
				return False

			unzipped_data = ZipFile(BytesIO(zip_data.content))
			unzipped_data.extractall(self.ticker_interval_directory)

			create_file_checksum(self.ticker_interval_directory / (base_file_name + '.csv'))

			return True



	def download_binance_dataset(self):
		"""
		creates a new directory inside @directory named from the ticker and interval
		"""

		if not self.ticker_interval_directory.exists():
			os.mkdir(self.ticker_interval_directory)

		cache_file_path = self.ticker_interval_directory / "cache.json"
		cache = {}

		if cache_file_path.exists():
			with open(cache_file_path, 'r') as cache_file:
				cache = json.load(cache_file)

		cache['ignore_dates'] = cache.get('ignore_dates', [])

		d = self.default_date
		end_date = datetime.now(timezone.utc).date() - timedelta(1)

		while d != end_date:

			date_str_day = d.strftime('%Y-%m-%d')
			date_str_month = d.strftime('%Y-%m')
			csv_file_path_day = self.ticker_interval_directory / f'{self.ticker_interval_id}-{date_str_day}.csv'
			csv_file_path_month = self.ticker_interval_directory / f'{self.ticker_interval_id}-{date_str_month}.csv'

			# monthly files are not availaible for the current month
			if d.day == 1 and (d.year != end_date.year or d.month != end_date.month):

				if verify_file_checksum(csv_file_path_month) or self.download_csv_file(date_str_month, True):				
					d = incr_month(d)
					continue

			if date_str_day not in cache['ignore_dates']:

				if not verify_file_checksum(csv_file_path_day) and not self.download_csv_file(date_str_day, False):
					cache['ignore_dates'].append(date_str_day) 
				
			d += timedelta(1)


		with open(cache_file_path, 'w') as cache_file:
			json.dump(cache, cache_file)


	def load_csv_files(self) -> pd.DataFrame:

		d = self.default_date
		end_date = datetime.now(timezone.utc).date() - timedelta(1)

		data_list = []
		columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'count', 'taker_buy_volume']

		while d != end_date:

			date_str_day = d.strftime('%Y-%m-%d')
			date_str_month = d.strftime('%Y-%m')
			csv_file_path_day = self.ticker_interval_directory / f'{self.ticker_interval_id}-{date_str_day}.csv'
			csv_file_path_month = self.ticker_interval_directory / f'{self.ticker_interval_id}-{date_str_month}.csv'

			# monthly files are not availaible for the current month
			if d.day == 1 and (d.year != end_date.year or d.month != end_date.month):

				if csv_file_path_month.exists():
					data_list.append(pd.read_csv(csv_file_path_month, header=None, index_col=0, usecols=[0,1,2,3,4,5,8,9], names=columns))
					d = incr_month(d)
					continue

			if csv_file_path_day.exists():
				data_list.append((pd.read_csv(csv_file_path_day, header=None, index_col=0, usecols=[0,1,2,3,4,5,8,9], names=columns)))

			d += timedelta(1)


		data = pd.concat(data_list)
		data.index = pd.to_datetime(data.index, unit='ms', origin='unix')

		return data


	def preprocess_binance_dataset(self):
		"""
		filters a dataset previously loaded
		"""

		cache_file_path = self.ticker_interval_directory / 'cache.json'
		cache = {}
		end_date = datetime.now(timezone.utc).date() - timedelta(1)

		store_path = self.ticker_interval_directory / 'data.h5'

		if cache_file_path.exists():
			with open(cache_file_path, 'r') as cache_file:
				cache = json.load(cache_file)
			
		if store_path.exists() and 'end_date' in cache and datetime.strptime(cache['end_date'], '%Y-%m-%d').date() == end_date:
			store = pd.HDFStore(store_path, 'r')
			preprocessed_chunks = [store.get(key) for key in sorted(store.keys()) if key.startswith('/chunk')]
			mean = store.get('mean')
			std = store.get('std')
			store.close()
			return preprocessed_chunks, mean, std
		

		data = self.load_csv_files()

		data = data.mask(data == 0).interpolate('time')

		data = data.asfreq(timedelta(minutes=1)).interpolate('linear', limit=self.max_gap_size)

		chunks_index = data.isna().any(axis=1).diff().fillna(0).cumsum()

		chunks: list[pd.DataFrame] = []

		groups = data.groupby(chunks_index)
		for g in islice(groups.groups, 0, None, 2):
			chunk = groups.get_group(g).dropna()
			if chunk.shape[0] > self.max_gap_size:
				chunks.append(chunk)

		
		for i, chunk in enumerate(chunks):
			# get log pct change for the price columns
			chunks[i][:] = np.log1p(chunk.pct_change())
			chunks[i].dropna(inplace=True)


		concatenated_chunks = pd.concat(chunks) 
		mean = concatenated_chunks.mean()
		std = concatenated_chunks.std()

		for i, chunk in enumerate(chunks):
			chunks[i] = (chunk - mean) / std

		store = pd.HDFStore(store_path, 'w')
		store.put('mean', mean)
		store.put('std', mean)
		for i, chunk in enumerate(chunks):
			store.put(f'chunk{i:03d}', chunk)
		store.close()

		cache['end_date'] = end_date.strftime('%Y-%m-%d')
		with open(cache_file_path, 'w') as cache_file:
			json.dump(cache, cache_file)

		return chunks, mean, std