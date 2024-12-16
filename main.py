import os
import copy
import utils
from utils import serialize
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold

_NUM_QUERY = 5 # Number of ensembles
_SHOT = 4 # Number of training shots
_SEED = 0 # Seed for fixing randomness
_DATA = 'myocardial'
_API_KEY = os.environ['OPENAI_API_KEY']
_TOGETHER_API_KEY = os.environ['TOGETHER_API_KEY']

#_DATA: 'diabetes', 'adult', 'bank', 'credit-g', 'car', 'heart', 'myocardial', 'communities'
#MODELS: "gpt-3.5-turbo-0125", "gpt-4-turbo-2024-04-09", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

utils.set_seed(_SEED)
df, X_train, X_test, y_train, y_test, target_attr, label_list, is_cat = utils.get_dataset(_DATA, _SHOT, _SEED)
X_all = df.drop(target_attr, axis=1)
print("dataset sample")
print(X_train.head(2))

ask_file_name = './templates/ask_llm_baseline.txt'
meta_data_name = f"./data/{_DATA}-metadata.json"
templates, feature_desc = utils.get_prompt_for_asking(
    _DATA, X_all, X_train, y_train, label_list, target_attr, ask_file_name, 
    meta_data_name, is_cat, num_query=_NUM_QUERY
)

# Feature subset selection
if len(X_test.columns) >= 20:
    total_column_list = []
    for i in range(len(X_test.columns) // 10):
        column_list = X_test.columns.tolist()
        random.shuffle(column_list)
        total_column_list.append(column_list[i*10:(i+1)*10])
else:
    total_column_list = [X_test.columns.tolist()]

total = 0
correct = 0

for package in tqdm(zip(X_test.iterrows(), y_test), total=len(X_test)):
    icl_idx, icl_row = package[0][0], package[0][1]
    gt_val = package[1]
    selected_column = random.choice(total_column_list)
    test_row = ""
    answer = gt_val
    icl_row = icl_row[selected_column]
    test_row += serialize(icl_row)
    test_row += f"\nAnswer: "
    curr_prompt = templates[0] + test_row
    # response = utils.query_gpt([curr_prompt], _API_KEY, max_tokens=1, temperature=0.1, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", type="together")
    response = utils.query_gpt([curr_prompt], _API_KEY, max_tokens=1, temperature=0.1, model="gpt-4-turbo-2024-04-09")
    print(response, gt_val)
    total += 1
    if response[0].lower() == gt_val.lower():
        correct += 1

print(f"Correct: {correct}/{total} ({correct/total})")
