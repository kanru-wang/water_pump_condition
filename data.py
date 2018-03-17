import pandas as pd
import numpy as np
import re
import math
from string import Template
from datetime import datetime
from pandas import datetime as dt
import warnings
import copy


class ClassificationData:
    data_attr = ('train_feature_df', 'train_outcome_df', 'test_feature_df')

    def __len__(self):
        return len(self.outcome_df)

    def __init__(self, train_feature_df, train_outcome_df, test_feature_df):
        self.train_feature_df = train_feature_df
        self.train_outcome_df = train_outcome_df
        self.test_feature_df = test_feature_df

    def __getitem__(self, postition):
        return ClassificationData(
            **{attr: getattr(self, attr)[postition]
               for attr in self.data_attr})

    def partition(self, num_partitions, p=None):
        if p and (sum(p) != 1.0):
            warnings.warn("Probabilities don't sum to 1, normalising them.")
            p = [pi / sum(p) for pi in p]

        index_labels = list(range(num_partitions))
        indices = np.random.choice(a=index_labels, size=len(self), p=p)

        partitions = [self[indices == il] for il in index_labels]
        return partitions
    

def cols_to_numeric(df, cols):
    df[cols] = df[cols].apply(lambda x: pd.to_numeric(x))
    return df


def cleaner(df):
    col_str, col_other = get_col_type_lists(df)
    return pd.concat(
        [df_other_cleaner(df[col_other]),
         df_string_cleaner(df[col_str])],
        axis=1)

# Need to transpose, get rid of the duplicated columns, and transpose again.
# Two ways to dedup: the current way (see below) or df_dummified_transposed.drop_duplicates().T
# https://stackoverflow.com/questions/13035764/remove-rows-with-duplicate-indices-pandas-dataframe-and-timeseries
def df_string_cleaner(df):
    df_clean = (df.fillna('undefined').applymap(clean_strings)
                .applymap(lambda x: x.lower()))
    #return pd.get_dummies(df_clean, drop_first = True)
    #return pd.get_dummies(df_clean)
    df_dummified_transposed  = pd.get_dummies(df_clean).T
    return df_dummified_transposed[~df_dummified_transposed.duplicated()].T


def df_other_cleaner(df):
    def x1(x):
        return x * 1.0

    #return df.fillna(0).apply(x1)
    return df.apply(x1)


def get_col_type_lists(df):
    # Build a pandas series with values indicating if column is string
    #   and axis labels giving column name.
    cs = df.applymap(is_numeric).apply(min)
    # (c)olumn (s)tring (l)ist
    csl = cs[~cs.values].axes[0].tolist()
    # (c)olumn (o)ther (l)ist
    col = cs[cs.values].axes[0].tolist()
    return csl, col


def is_numeric(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def boolean_to_numeric(dfr, col):
    df = copy.deepcopy(dfr)
    df[col] = df[col].apply(lambda x: 1.0 if x.lower() == 'true' else 0.0)
    return df 


def clean_strings(s):
    '''
    The output from the PMML will be put into C# so we need to ensure the
    strings are compatible
    '''
    s_tmp = re.sub("\s+", '_', s)
    s_clean = '_' + \
        s_tmp.replace("'", '')\
        .replace('&', 'and')\
        .replace('-', '_')
    return s_clean


def measure_length(x):
    try:
        length = len(x)
    except TypeError:
        length = np.nan
    return length


def group_small_levels(df, target_col_name, freq_col_name, threshold):
    return(df.apply(lambda x: x[target_col_name] 
                    if x[freq_col_name] >= threshold 
                    else 'small_levels', 
                    axis=1))
    