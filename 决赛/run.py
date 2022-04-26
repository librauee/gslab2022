import re
import os
import sklearn
import json
import pandas as pd
import warnings

import lightgbm as lgb
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, fbeta_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

warnings.filterwarnings('ignore')


train_column_name = '''worldid
iuin
dtgamestarttime
iduration
igameseq
isubgamemode
irankingame
idamgedealt
idamagetaken
igoldearned
ihealingdone
ilargestkillingspree
ilargestmultikill
imagicdamagedealt
imagicdamagetaken
iminionskilled
ineutralmonsterkills
iphysicaldamagedealt
iphysicaldamagetaken
champion_id
premade_size
elo_change
team
champions_killed
num_deaths
assists
game_score
flag
ext_flag
champion_used_exp
spell1
spell2
win
tier
queue'''.split('\n')


def reduce_and_save(path, usecols, dtype):
    train = pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype,
        sep='|',
        header=None,
        names=train_column_name,
        skiprows=[104341489, 148848549, 189866023]
    )
    train.to_pickle('train.pkl')


path = 'MOBA游戏对局胜负预测赛题/battle_data_one_month.txt'
dtype = {
    'dtgamestarttime': 'str',
    'elo_change': 'uint8',

    'iduration': 'uint16',
    'idamgedealt': 'uint16',
    'idamagetaken': 'uint16',
    'igoldearned': 'uint16',
    'ihealingdone': 'uint16',
    'ilargestkillingspree': 'uint8',
    'ilargestmultikill': 'uint8',
    'imagicdamagedealt': 'uint16',
    'imagicdamagetaken': 'uint16',
    'iminionskilled': 'uint16',
    'ineutralmonsterkills': 'uint8',
    'iphysicaldamagedealt': 'uint16',
    'iphysicaldamagetaken': 'uint16',

    'isubgamemode': 'uint8',
    'iuin': 'int64',
    'igameseq': 'int64',
    'isubgamemode': 'uint8',
    'irankingame': 'uint16',
    'champion_id': 'uint16',
    'premade_size': 'uint8',
    'team': 'uint8',
    'spell1': 'uint8',
    'spell2': 'uint8',
    'win': 'uint8',
    'tier': 'uint8',
    'queue': 'uint8',

    'champions_killed': 'uint8',
    'num_deaths': 'uint8',
    'assists': 'uint8',
    'game_score': 'uint16',
    'flag': 'uint8',
    'ext_flag': 'uint32',
    'champion_used_exp': 'uint16',
}
features = dtype.keys()

if not os.path.exists('train.pkl'):
    reduce_and_save(path, features, dtype=dtype)
    data = pd.read_pickle('train.pkl')
else:
    data = pd.read_pickle('train.pkl')

data['dtgamestarttime'] = pd.to_datetime(data['dtgamestarttime'])
data = data[(data['isubgamemode'] == 20) & (data['tier'] == 255) & (data['queue'] == 255) & (data['elo_change'] != 0) & (data['iduration'] > 300)]
data['KDA'] = (data['champions_killed'] + data['assists']) / (data['num_deaths'] + 1e-3)
data['ks'] = data['iminionskilled'] / data['iduration']
print(len(data))

test = pd.read_csv('MOBA游戏对局胜负预测赛题/battle_data_one_day.txt', sep='|', header=None)
test.columns = '''worldid
iuin
dtgamestarttime
igameseq
isubgamemode
irankingame
champion_id
premade_size
team
champion_used_exp
spell1
spell2
tier
queue'''.split('\n')


CV = False

if CV:
    train = data[(data['dtgamestarttime'] > '2020-3-30') & (data['dtgamestarttime'] < '2020-3-31')]
    val = data[(data['dtgamestarttime'] >= '2020-3-31')]
    print(len(train))
    print(len(val))
    train['win_team'] = np.where((train['win']==1) & (train['team'] == 100) | ((train['win']==0) & (train['team'] == 200)), 100, 200)
    train_rank = train.drop_duplicates(['igameseq'])[['igameseq', 'win_team']]
    train_rank['label'] = train_rank['win_team'].apply(lambda x: 1 if x == 100 else 0)
    print(len(train_rank))
    val['win_team'] = np.where((val['win']==1) & (val['team'] == 100) | ((val['win']==0) & (val['team'] == 200)), 100, 200)
    val_rank = val.drop_duplicates(['igameseq'])[['igameseq', 'win_team']]
    val_rank['label'] = val_rank['win_team'].apply(lambda x: 1 if x == 100 else 0)
    print(len(val_rank))
else:
    train = data[(data['dtgamestarttime'] > '2020-3-27') & (data['dtgamestarttime'] < '2020-4-1')]
    train['win_team'] = np.where((train['win']==1) & (train['team'] == 100) | ((train['win']==0) & (train['team'] == 200)), 100, 200)
    train_rank = train.drop_duplicates(['igameseq'])[['igameseq', 'win_team']]
    train_rank['label'] = train_rank['win_team'].apply(lambda x: 1 if x == 100 else 0)
    print(len(train))
    print(len(train_rank))
    val = test.copy()
    print(len(val))
    val_rank = test.drop_duplicates(['igameseq'])[['igameseq']]
    print(len(val_rank))


# # 历史胜率

train['winning_rate_iuin'] = train.groupby(['iuin'])['win'].transform('mean')
train['winning_rate_champion'] = train.groupby(['champion_id'])['win'].transform('mean')

val = pd.merge(val, train[['iuin', 'winning_rate_iuin']].drop_duplicates(), on='iuin', how='left')
val = pd.merge(val, train[['champion_id', 'winning_rate_champion']].drop_duplicates(), on='champion_id', how='left')

### 按玩家聚合

train_iuin = train.drop_duplicates(['iuin'])[['iuin']]
GAME_FEATURES = [
    'idamgedealt',
    'idamagetaken',
    'igoldearned',
    'ihealingdone',
    'ilargestkillingspree',
    'ilargestmultikill',
    'imagicdamagedealt',
    'imagicdamagetaken',
    'iminionskilled',
    'ineutralmonsterkills',
    'iphysicaldamagedealt',
    'iphysicaldamagetaken',
    'champions_killed',
    'num_deaths',
    'assists',
    'game_score',
    'KDA',
    'ks',
]

for f in tqdm(GAME_FEATURES):
    t = train.groupby(['iuin'])[f].agg([
        ('{}_mean_iuin'.format(f), 'mean'),
        ('{}_std_iuin'.format(f), 'std'),
        ('{}_max_iuin'.format(f), 'max'),
        ('{}_min_iuin'.format(f), 'min'),
    ]).reset_index()

    train_iuin = pd.merge(train_iuin, t, on='iuin', how='left')

# 按英雄聚合

train_champion = train.drop_duplicates(['champion_id'])[['champion_id']]

for f in tqdm(GAME_FEATURES):
    t = train.groupby(['champion_id'])[f].agg([
        ('{}_mean_champion'.format(f), 'mean'),
        ('{}_std_champion'.format(f), 'std'),
        ('{}_max_champion'.format(f), 'max'),
        ('{}_min_champion'.format(f), 'min'),
    ]).reset_index()

    train_champion = pd.merge(train_champion, t, on='champion_id', how='left')

# 按玩家-英雄聚合

train_champion_iuin = train.drop_duplicates(['champion_id', 'iuin'])[['champion_id', 'iuin']]

for f in tqdm(GAME_FEATURES):
    t = train.groupby(['champion_id', 'iuin'])[f].agg([
        ('{}_mean_champion_iuin'.format(f), 'mean'),
        ('{}_std_champion_iuin'.format(f), 'std'),
        ('{}_max_champion_iuin'.format(f), 'max'),
        ('{}_min_champion_iuin'.format(f), 'min'),
    ]).reset_index()

    train_champion_iuin = pd.merge(train_champion_iuin, t, on=['champion_id', 'iuin'], how='left')

iuin_features = [i for i in train_iuin.columns if i != 'iuin']
train = pd.merge(train, train_iuin, on='iuin', how='left')
champion_features = [i for i in train_champion.columns if i != 'champion_id']
train = pd.merge(train, train_champion, on='champion_id', how='left')
champion_iuin_features = [i for i in train_champion_iuin.columns if i not in ['champion_id', 'iuin']]
train = pd.merge(train, train_champion_iuin, on=['champion_id', 'iuin'], how='left')

train_100 = train[train['team'] == 100]
train_200 = train[train['team'] == 200]

# 分队伍
for f in tqdm(iuin_features + champion_features):
    t = train_100.groupby(['igameseq'])[f].agg([
        ('{}_mean_team_red'.format(f), 'mean'),
        ('{}_std_team_red'.format(f), 'std'),
        ('{}_max_team_red'.format(f), 'max'),
        ('{}_min_team_red'.format(f), 'min'),
    ]).reset_index()

    train_rank = pd.merge(train_rank, t, on='igameseq', how='left')

    t = train_200.groupby(['igameseq'])[f].agg([
        ('{}_mean_team_blue'.format(f), 'mean'),
        ('{}_std_team_blue'.format(f), 'std'),
        ('{}_max_team_blue'.format(f), 'max'),
        ('{}_min_team_blue'.format(f), 'min'),
    ]).reset_index()

    train_rank = pd.merge(train_rank, t, on='igameseq', how='left')

    train_rank[f'{f}_diff1'] = train_rank[f'{f}_min_team_red'] - train_rank[f'{f}_min_team_blue']
    train_rank[f'{f}_diff2'] = train_rank[f'{f}_max_team_red'] - train_rank[f'{f}_max_team_blue']
    train_rank[f'{f}_diff3'] = train_rank[f'{f}_max_team_red'] - train_rank[f'{f}_min_team_blue']
    train_rank[f'{f}_diff4'] = train_rank[f'{f}_min_team_red'] - train_rank[f'{f}_max_team_blue']
    train_rank[f'{f}_diff5'] = train_rank[f'{f}_mean_team_red'] - train_rank[f'{f}_mean_team_blue']
    train_rank[f'{f}_diff6'] = train_rank[f'{f}_std_team_red'] - train_rank[f'{f}_std_team_blue']

val = pd.merge(val, train_iuin, on='iuin', how='left')
val = pd.merge(val, train_champion, on='champion_id', how='left')
val = pd.merge(val, train_champion_iuin, on=['champion_id', 'iuin'], how='left')

val_100 = val[val['team'] == 100]
val_200 = val[val['team'] == 200]

# 分队伍
for f in tqdm(iuin_features + champion_features):
    t = val_100.groupby(['igameseq'])[f].agg([
        ('{}_mean_team_red'.format(f), 'mean'),
        ('{}_std_team_red'.format(f), 'std'),
        ('{}_max_team_red'.format(f), 'max'),
        ('{}_min_team_red'.format(f), 'min'),
    ]).reset_index()

    val_rank = pd.merge(val_rank, t, on='igameseq', how='left')

    t = val_200.groupby(['igameseq'])[f].agg([
        ('{}_mean_team_blue'.format(f), 'mean'),
        ('{}_std_team_blue'.format(f), 'std'),
        ('{}_max_team_blue'.format(f), 'max'),
        ('{}_min_team_blue'.format(f), 'min'),
    ]).reset_index()

    val_rank = pd.merge(val_rank, t, on='igameseq', how='left')

    val_rank[f'{f}_diff1'] = val_rank[f'{f}_min_team_red'] - val_rank[f'{f}_min_team_blue']
    val_rank[f'{f}_diff2'] = val_rank[f'{f}_max_team_red'] - val_rank[f'{f}_max_team_blue']
    val_rank[f'{f}_diff3'] = val_rank[f'{f}_max_team_red'] - val_rank[f'{f}_min_team_blue']
    val_rank[f'{f}_diff4'] = val_rank[f'{f}_min_team_red'] - val_rank[f'{f}_max_team_blue']
    val_rank[f'{f}_diff5'] = val_rank[f'{f}_mean_team_red'] - val_rank[f'{f}_mean_team_blue']
    val_rank[f'{f}_diff6'] = val_rank[f'{f}_std_team_red'] - val_rank[f'{f}_std_team_blue']


# 某场比赛  总ELO、人数、胜率聚合
for f in tqdm(['irankingame', 'premade_size',
               'winning_rate_iuin', 'winning_rate_champion'
              ]):
    t = train.groupby(['igameseq'])[f].agg([
            ('{}_mean'.format(f), 'mean'),
            ('{}_std'.format(f), 'std'),
            ('{}_max'.format(f), 'max'),
            ('{}_min'.format(f), 'min'),
        ]).reset_index()

    train_rank = pd.merge(train_rank, t, on='igameseq', how='left')

    # 分队伍

    t = train_100.groupby(['igameseq'])[f].agg([
            ('{}_mean_team_red'.format(f), 'mean'),
            ('{}_std_team_red'.format(f), 'std'),
            ('{}_max_team_red'.format(f), 'max'),
            ('{}_min_team_red'.format(f), 'min'),
        ]).reset_index()

    train_rank = pd.merge(train_rank, t, on='igameseq', how='left')


    t = train_200.groupby(['igameseq'])[f].agg([
            ('{}_mean_team_blue'.format(f), 'mean'),
            ('{}_std_team_blue'.format(f), 'std'),
            ('{}_max_team_blue'.format(f), 'max'),
            ('{}_min_team_blue'.format(f), 'min'),
        ]).reset_index()

    train_rank = pd.merge(train_rank, t, on='igameseq', how='left')

    train_rank[f'{f}_diff1'] = train_rank[f'{f}_min_team_red'] - train_rank[f'{f}_min_team_blue']
    train_rank[f'{f}_diff2'] = train_rank[f'{f}_max_team_red'] - train_rank[f'{f}_max_team_blue']
    train_rank[f'{f}_diff3'] = train_rank[f'{f}_max_team_red'] - train_rank[f'{f}_min_team_blue']
    train_rank[f'{f}_diff4'] = train_rank[f'{f}_min_team_red'] - train_rank[f'{f}_max_team_blue']
    train_rank[f'{f}_diff5'] = train_rank[f'{f}_mean_team_red'] - train_rank[f'{f}_mean_team_blue']
    train_rank[f'{f}_diff6'] = train_rank[f'{f}_std_team_red'] - train_rank[f'{f}_std_team_blue']


    t = val.groupby(['igameseq'])[f].agg([
            ('{}_mean'.format(f), 'mean'),
            ('{}_std'.format(f), 'std'),
            ('{}_max'.format(f), 'max'),
            ('{}_min'.format(f), 'min'),
        ]).reset_index()

    val_rank = pd.merge(val_rank, t, on='igameseq', how='left')

    # 分队伍

    t = val_100.groupby(['igameseq'])[f].agg([
            ('{}_mean_team_red'.format(f), 'mean'),
            ('{}_std_team_red'.format(f), 'std'),
            ('{}_max_team_red'.format(f), 'max'),
            ('{}_min_team_red'.format(f), 'min'),
        ]).reset_index()

    val_rank = pd.merge(val_rank, t, on='igameseq', how='left')


    t = val_200.groupby(['igameseq'])[f].agg([
            ('{}_mean_team_blue'.format(f), 'mean'),
            ('{}_std_team_blue'.format(f), 'std'),
            ('{}_max_team_blue'.format(f), 'max'),
            ('{}_min_team_blue'.format(f), 'min'),
        ]).reset_index()

    val_rank = pd.merge(val_rank, t, on='igameseq', how='left')

    val_rank[f'{f}_diff1'] = val_rank[f'{f}_min_team_red'] - val_rank[f'{f}_min_team_blue']
    val_rank[f'{f}_diff2'] = val_rank[f'{f}_max_team_red'] - val_rank[f'{f}_max_team_blue']
    val_rank[f'{f}_diff3'] = val_rank[f'{f}_max_team_red'] - val_rank[f'{f}_min_team_blue']
    val_rank[f'{f}_diff4'] = val_rank[f'{f}_min_team_red'] - val_rank[f'{f}_max_team_blue']
    val_rank[f'{f}_diff5'] = val_rank[f'{f}_mean_team_red'] - val_rank[f'{f}_mean_team_blue']
    val_rank[f'{f}_diff6'] = val_rank[f'{f}_std_team_red'] - val_rank[f'{f}_std_team_blue']

import catboost

features = [i for i in train_rank.columns if i not in ['label', 'igameseq', 'win_team']]

print("Train files: ", len(train_rank), "| Test files: ", len(val_rank), "| Feature nums", len(features))

if CV:
    cat_params = {
        'eval_metric': 'Accuracy',
        'random_seed': 666,
        'logging_level': 'Verbose',
        'use_best_model': True,
        'loss_function': 'Logloss',
        'task_type': 'GPU',
        'learning_rate': 0.1
    }
    trn_data = catboost.Pool(train_rank[features], label=train_rank['label'])
    val_data = catboost.Pool(val_rank[features], label=val_rank['label'])
    num_round = 10000
    clf = catboost.train(
        params=cat_params,
        pool=trn_data,
        iterations=num_round,
        eval_set=val_data,
        verbose_eval=100,
        early_stopping_rounds=50,
    )

    oof = [i[1] for i in clf.predict(val_rank[features], prediction_type='Probability')]

    print("Accuracy score: {}".format(accuracy_score(val_rank['label'], [1 if i > 0.5 else 0 for i in oof])))

else:
    cat_params = {
        'eval_metric': 'Accuracy',
        'random_seed': 666,
        'logging_level': 'Verbose',
        'loss_function': 'Logloss',
        'task_type': 'GPU',
        'learning_rate': 0.1
    }

    trn_data = catboost.Pool(train_rank[features], label=train_rank['label'])
    num_round = 300
    clf = catboost.train(
        params=cat_params,
        pool=trn_data,
        iterations=num_round,
        verbose_eval=50,

    )

    oof = [i[1] for i in clf.predict(val_rank[features], prediction_type='Probability')]

    val_rank['gameid'] = list(val_rank['igameseq'])
    val_rank['teamid'] = 100
    val_rank['result'] = [1 if i > 0.5 else 0 for i in oof]
    val_rank[['igameseq', 'teamid', 'result']].to_csv('result.csv', index=False)
    val_rank[['igameseq', 'teamid', 'result']].head()
