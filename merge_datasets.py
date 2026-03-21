import os
import numpy as np
import pandas as pd

dataset_list = ["DoubanMusic", "DoubanMovie", "DoubanBook"]


for idx, data in enumerate(dataset_list):
    df = pd.read_csv("./dataset/%s/%s.inter" % (data, data), delimiter="\t")[["user_id:token", "item_id:token", "rating:float", "timestamp:float"]]
    df["item_type:token"] = idx
    df.to_csv("./dataset/%s/%s.inter" % (data, data), sep='\t', index=False)
    # df = pd.read_csv("./dataset/%s/%s.item" % (data, data), delimiter="\t")
    
    # df.to_csv("./dataset/%s/%s.item" % (data, data), sep='\t', index=False)
    
    df_item = pd.DataFrame()
    df_item["item_id:token"] = df["item_id:token"].unique()
    df_item["item_type:token"] = idx
    df_item.to_csv("./dataset/%s/%s.item" % (data, data), sep='\t', index=False)


def merge_datasets_inter(dataset_list):
    path_list = ["./dataset/%s/%s.inter" % (dataset, dataset) for dataset in dataset_list]
    L = []
    for idx, path in enumerate(path_list):
        df = pd.read_csv(path, delimiter="\t")
        L.append(df)
    return pd.concat(L)


dataset_list = ["DoubanMusic", "DoubanMovie"]
merge_name = "DoubanAll_woT"
merge_df_inter = merge_datasets_inter(dataset_list)


if not os.path.exists("./dataset/%s" % merge_name):
    os.mkdir("./dataset/%s" % merge_name)
merge_df_inter.to_csv("./dataset/%s/%s.inter" % (merge_name, merge_name), sep='\t', index=False)
