from typing import Tuple, List
import yaml
import numpy
import pandas as pd
from src.feature import ngram
import lightgbm as lgb
from util.util import *

config = yaml.load(open('config/config.yaml'), Loader=yaml.FullLoader)
feature_func_dict = {
    'ngram': ngram
}
assert config['feature']['selected'] in feature_func_dict
assert config['model']['selected'] == 'lightgbm'
feature_func = feature_func_dict[config['feature']['selected']]

def train_test() -> None:
    print('Read data features from both train and test data')
    train_data, test_data = feature_func()
    print('LightGBM training...')
    lgb_model = lgb.train(
        train_set=lgb.Dataset(data=train_data.iloc[:, 3:], label=train_data['score']),
        num_boost_round=config['model']['hyperameters']['lightgbm']['custom']['num_boost_round'], 
        params=config['model']['hyperameters']['lightgbm']['built_in']
    )
    print('Saving the result...')
    res_table = test_data.loc[:, ['id']]
    res_table['delta_g'] = test_data['delta_g'] = lgb_model.predict(
        data=test_data.iloc[:, 3:]
    )
    result_dir = make_result_dir()
    res_table.to_csv(os.path.join(result_dir, 'result.csv'), index=False)

def train_val() -> None:
    pass

def train_kfold() -> None:
    pass



if __name__ == '__main__':
    train_type_func = {
        'train_kfold': train_kfold,
        'train_val': train_val,
        'train_test': train_test
    }
    train_type = config['train']['selected']
    train_type_func[train_type]()

