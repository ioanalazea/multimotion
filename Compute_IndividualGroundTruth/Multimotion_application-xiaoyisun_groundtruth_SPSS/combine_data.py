# -*- coding: utf-8 -*-
# @Time    : 2023/8/2 16:11
# @Author  : Xiaoyi Sun
# @Site    : 
# @File    : combine_data.py
# @Software: PyCharm


import pandas as pd


def concatenate_dataframes(new_dataframe_path, new_dataframe_path_2, updated_dataframe):
    new_df = pd.read_csv(new_dataframe_path)
    new_df_2 = pd.read_csv(new_dataframe_path_2)
    updated_df = pd.concat([new_df, new_df_2])
    updated_df.to_csv(updated_dataframe, index=False)

