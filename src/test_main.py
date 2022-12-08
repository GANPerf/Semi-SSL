import pandas as pd

from src.main import is_pick_unlabeled_data

data =pd.read_csv('/data/huawei/Semi-SSL/CUB200/test_loop.csv')

confidence_unlabeled=0.95
is_pick_unlabeled_data(data,confidence_unlabeled)