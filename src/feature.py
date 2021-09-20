from typing import Tuple, List
import yaml
import numpy
import pandas as pd
from tqdm import tqdm

"""
Reference from https://github.com/ji1ai1/202109-QHL/
"""

config = yaml.load(open('config/config.yaml'), Loader=yaml.FullLoader)

def ngram() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Read train table and test table
    train_table, test_table = _load_train_test_data()
    
    # get ngram from train table.
    print('Get ngram from train table.')
    train_base_feature_table = train_table.loc[:, ["id", "antibody_seq_a", "antibody_seq_b", "antigen_seq"]]
    for field in ["antibody_seq_a", "antibody_seq_b", "antigen_seq"]:
        train_base_feature_table["%s_len" % field] = train_base_feature_table[field].str.len()
        for c1 in tqdm([chr(65 + k) for k in range(26)]):
            train_base_feature_table["%s_%s" % (field, c1)] = train_base_feature_table[field].str.count(c1)
            for c2 in [chr(65 + k) for k in range(26)]:
                train_base_feature_table["%s_%s" % (field, c1 + c2)] = train_base_feature_table[field].str.count(c1 + c2)
                for c3 in [chr(65 + k) for k in range(26)]:
                    train_base_feature_table["%s_%s" % (field, c1 + c2 + c3)] = train_base_feature_table[field].str.count(c1 + c2 + c3)
    train_base_feature_table = train_base_feature_table.drop(["antibody_seq_a", "antibody_seq_b", "antigen_seq"], axis=1)

    # get Bow from test table.
    print('Get ngram from test table.')
    test_base_feature_table = test_table.loc[:, ["id", "antibody_seq_a", "antibody_seq_b", "antigen_seq"]]
    for field in ["antibody_seq_a", "antibody_seq_b", "antigen_seq"]:
        test_base_feature_table["%s_len" % field] = test_base_feature_table[field].str.len()
        for c1 in tqdm([chr(65 + k) for k in range(26)]):
            test_base_feature_table["%s_%s" % (field, c1)] = test_base_feature_table[field].str.count(c1)
            for c2 in [chr(65 + k) for k in range(26)]:
                test_base_feature_table["%s_%s" % (field, c1 + c2)] = test_base_feature_table[field].str.count(c1 + c2)
                for c3 in [chr(65 + k) for k in range(26)]:
                    test_base_feature_table["%s_%s" % (field, c1 + c2 + c3)] = test_base_feature_table[field].str.count(c1 + c2 + c3)
    test_base_feature_table = test_base_feature_table.drop(["antibody_seq_a", "antibody_seq_b", "antigen_seq"], axis=1)

    # K折获取训练数据表
    K = config['feature']['hyperameters']['ngram']['kfold']
    train_table_list = []
    for k in range(K):
        candidate_table = train_table.iloc[[idx for idx in range(len(train_table)) if idx % K == k]].reset_index(drop=True)
        statistics_table = train_table.iloc[[idx for idx in range(len(train_table)) if idx % K != k]].reset_index(drop=True)
        train_table_list.append(
            _extract_from_base_features(candidate_table, train_base_feature_table, statistics_table)
        )
    res_train_table = pd.concat(train_table_list, ignore_index=True)

    # 获取测试数据表
    res_test_table = _extract_from_base_features(test_table, test_base_feature_table, train_table)
    
    return res_train_table, res_test_table

def _feature_statistics(table: pd.DataFrame, keys: List[str], statistics_dict: dict, prefix: str = '') -> pd.DataFrame:
    if not isinstance(keys, list):
        keys = [keys]
    statistics_table = table.groupby(keys).aggregate(statistics_dict)
    statistics_table.columns = ["%s%s_%s%s" % (prefix, "".join(keys), field, aggfun if isinstance(aggfun, str) else aggfun.__name__) for field, aggfun_list in statistics_dict.items() for aggfun in (aggfun_list if isinstance(aggfun_list, list) else [aggfun_list])]
    return statistics_table

def _extract_from_base_features(table, base_feature_table, statistics_table):
   res_table = table
   res_table = res_table.merge(base_feature_table, on="id", how="left")
   res_table = res_table.merge(_feature_statistics(statistics_table, "antibody_seq_a", {"delta_g": ["mean", "median", "min", "max"]}).reset_index(), on="antibody_seq_a", how="left")
   res_table = res_table.merge(_feature_statistics(statistics_table, "antibody_seq_b", {"delta_g": ["mean", "median", "min", "max"]}).reset_index(), on="antibody_seq_b", how="left")
   res_table = res_table.merge(_feature_statistics(statistics_table, "antigen_seq", {"delta_g": ["mean", "median", "min", "max"]}).reset_index(), on="antigen_seq", how="left")
   res_table = res_table.drop(["pdb", "antibody_seq_a", "antibody_seq_b", "antigen_seq"], axis=1)
   
   res_table["score"] = res_table.delta_g.rank()
   res_table = res_table.loc[:, ["id", "delta_g", "score"] + [field for field in res_table.columns if field not in ["id", "delta_g", "score"]]]
   
   return res_table

def _load_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    从原始数据中读表
    """
    train_table = pd.read_csv(config['data']['train'], sep='\t')
    train_table['id'] = range(-1, -len(train_table)-1, -1)
    test_table = pd.read_csv(config['data']['testA'], sep='\t')
    test_table['delta_g'] = -1
    return train_table, test_table

