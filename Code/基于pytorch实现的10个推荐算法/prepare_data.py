import numpy as np
import pandas as pd

data_path = './data/movielens.txt'
samp_rows = 1000
data = pd.read_csv(data_path, nrows=samp_rows)

data = data.sort_values(by='timestamp', ascending=True)
train = data.iloc[:int(len(data)*0.8)].copy()
train['rating'] = train['rating'].apply(lambda x: 1 if x > 3 else 0)
test = data.iloc[int(len(data)*0.8):].copy()

valid = test[['user_id', 'movie_id', 'rating']].groupby('user_id').agg(list).reset_index()

print(1)