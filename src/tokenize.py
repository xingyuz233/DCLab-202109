
from typing import List, Tuple
import yaml
import numpy
import pandas as pd

config = yaml.load(open('config/config.yaml'), Loader=yaml.FullLoader)

def n_gram

def _load_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    从原始数据中读表
    """
    train_table = pd.read_csv(config['data']['train'], sep='\t')
    train_table['id'] = range(-1, -len(train_table)-1, -1)
    test_table = pd.read_csv(config['data']['test'], sep='\t')
    test_table['delta_g'] = -1
    return train_table, test_table

