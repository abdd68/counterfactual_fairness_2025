import numpy as np
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
import pandas as pd
import csv
import random
import category_encoders as ce
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE as tsn
from sklearn import metrics
import os
import urllib

font_sz = 24
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
matplotlib.rcParams.update({'font.size': font_sz})

def load_data(path, name):
    print('loading dataset: ', name)
    if name == 'law':
        csv_data = pd.read_csv(path)

        # index selection
        selected_races = ['White', 'Black', 'Asian']
        print("select races: ", selected_races)
        select_index = np.array(csv_data[(csv_data['race'] == selected_races[0]) | (csv_data['race'] == selected_races[1]) |
                                         (csv_data['race'] == selected_races[2])].index, dtype=int)
        # shuffle
        np.random.shuffle(select_index)

        LSAT = csv_data[['LSAT']].to_numpy()[select_index]  # n x 1
        UGPA = csv_data[['UGPA']].to_numpy()[select_index]  # n x 1
        x = csv_data[['LSAT','UGPA']].to_numpy()[select_index]  # n x d
        ZFYA = csv_data[['ZFYA']].to_numpy()[select_index]  # n x 1
        sex = csv_data[['sex']].to_numpy()[select_index] - 1  # n x 1

        n = ZFYA.shape[0]
        rr = csv_data['race']
        env_race = csv_data['race'][select_index].to_list()  # n, string list
        env_race_id = np.array([selected_races.index(env_race[i]) for i in range(n)]).reshape(-1, 1)
        data_save = {'data': {'LSAT': LSAT, 'UGPA': UGPA, 'ZFYA': ZFYA, 'race': env_race_id, 'sex': sex}}

    if name == 'adult':
        csv_data = pd.read_csv(path)
        csv_data.replace(' ?', np.NaN, inplace=True)  # Replacing all the missing values with NaN

        csv_data.columns = ["Age", "WorkClass", "fnlwgt", "Education", "EducationNum", "MaritalStatus", "Occupation", "Relationship", "Race", "Sex",
                            "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"]  # for ease of human interpretation

        column_of_interest = ['Race', 'Sex', 'MaritalStatus', 'Occupation', 'EducationNum', 'HoursPerWeek', 'Income']
        column_of_interest_cat = ['Sex', 'MaritalStatus', 'Occupation']
        column_of_interest_num = ['EducationNum', 'HoursPerWeek']
        csv_data = csv_data[column_of_interest]

        # drop NaNs
        csv_data.dropna(axis=0, how='any', inplace=True)  # Dropping all the missing values (hence reduced training set)

        # index selection
        selected_races = ['White', 'Black', 'Asian-Pac-Islander']  # races = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
        print("select races: ", selected_races)
        csv_data['Race'] = csv_data['Race'].str.strip()
        csv_data = csv_data.loc[(csv_data['Race'] == selected_races[0]) | (csv_data['Race'] == selected_races[1]) | (csv_data['Race'] == selected_races[2])]

        Income = csv_data["Income"].map({' <=50K': 0, ' >50K': 1}).to_numpy().reshape(-1, 1)  # just to give binary labels
        csv_data.drop(["Income"], axis=1, inplace=True)

        n = len(csv_data.index)
        select_index = np.arange(n)
        np.random.shuffle(select_index)
        #
        env_race = csv_data['Race'].values[select_index]  # n, string list
        env_race_id = np.array([selected_races.index(env_race[i]) for i in range(n)]).reshape(-1, 1)

        data_save = {'data': {'Race': env_race_id, 'Income': Income[select_index]}}

        # categorical features: one hot
        for cat in column_of_interest_cat:
            encoding_pipeline = Pipeline([
                ('encode_cat', ce.OneHotEncoder(cols=cat, return_df=True))
            ])
            feat_onehot = encoding_pipeline.fit_transform(csv_data[[cat]]).values
            data_save['data'][cat] = feat_onehot[select_index]

        # numbers
        for nu in column_of_interest_num:
            data_save['data'][nu] = csv_data[nu].to_numpy()[select_index].reshape(-1, 1)
            
    if name == 'crimes':
        if not os.path.isfile('communities.data'):
            urllib.request.urlretrieve(
                "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data", "communities.data")
            urllib.request.urlretrieve(
                "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names",
                "communities.names")

        # create names
        names = []
        with open('communities.names', 'r') as file:
            for line in file:
                if line.startswith('@attribute'):
                    names.append(line.split(' ')[1])

        # load data
        data = pd.read_csv('communities.data', names=names, na_values=['?'])
        # Columns of interest
        column_of_interest = ['population', 'medIncome', 'PctPopUnderPov', 'PctUnemployed', 'PctNotHSGrad']
        label = 'ViolentCrimesPerPop'
        sensitive_attribute = 'racepctblack'

        # Filter the dataset
        data_filtered = data[column_of_interest + [label, sensitive_attribute]]

        # Process the sensitive attribute
        data_filtered[sensitive_attribute] = pd.cut(
            data_filtered[sensitive_attribute],
            bins=[-float('inf'), 0.33, 0.67, float('inf')],
            labels=[0, 1, 2]
        ).astype(int)

        # Convert data to numpy arrays and unsqueeze the last dimension
        data_save = {
            'data': {
                'population': data_filtered['population'].values[:, np.newaxis],
                'medIncome': data_filtered['medIncome'].values[:, np.newaxis],
                'PctPopUnderPov': data_filtered['PctPopUnderPov'].values[:, np.newaxis],
                'PctUnemployed': data_filtered['PctUnemployed'].values[:, np.newaxis],
                'PctNotHSGrad': data_filtered['PctNotHSGrad'].values[:, np.newaxis],
                'label': data_filtered[label].values[:, np.newaxis],
                'race': data_filtered[sensitive_attribute].values[:, np.newaxis]
            }
        }

    return data_save # x, y, env

def split_data(n, rates=[0.6, 0.2, 0.2], labels=None, type='random', sorted=False, label_number=1000000, seed=1):

    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    trn_rate, val_rate, tst_rate = rates[0], rates[1], rates[2]

    if type == 'ratio':  # follow the original ratio of label distribution, only applicable to binary classification!
        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]

        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)

        idx_train = np.append(label_idx_0[:min(int(trn_rate * len(label_idx_0)), label_number // 2)],
                                label_idx_1[:min(int(trn_rate * len(label_idx_1)), label_number // 2)])
        idx_val = np.append(label_idx_0[int(trn_rate * len(label_idx_0)):int((trn_rate + val_rate) * len(label_idx_0))],
                            label_idx_1[int(trn_rate * len(label_idx_1)):int((trn_rate + val_rate) * len(label_idx_1))])
        idx_test = np.append(label_idx_0[int((trn_rate + val_rate) * len(label_idx_0)):], label_idx_1[int((trn_rate + val_rate) * len(label_idx_1)):])

        np.random.shuffle(idx_train)
        np.random.shuffle(idx_val)
        np.random.shuffle(idx_test)

        if sorted:
            idx_train.sort()
            idx_val.sort()
            idx_test.sort()

    elif type == 'random':
        idx_all = np.arange(n)
        idx_train = np.random.choice(n, size=int(trn_rate * n), replace=False)
        idx_left = np.setdiff1d(idx_all, idx_train)
        idx_val = np.random.choice(idx_left, int(val_rate * n), replace=False)
        idx_test = np.setdiff1d(idx_left, idx_val)

        if sorted:
            idx_train.sort()
            idx_val.sort()
            idx_test.sort()
    # elif type == "balanced":

    return idx_train, idx_val, idx_test

def mmd_linear(X, Y, p=0.003):
    delta = (X.mean(0) - Y.mean(0))/p
    mmd = delta.dot(delta.T)
    return mmd

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)
    
def wasserstein(x, y, device, p=0.5, lam=10, its=10, sq=False, backpropT=False, cuda=False, scal=0.005):
    """return W dist between x and y"""
    '''distance matrix M'''

    nx = x.shape[0]
    ny = y.shape[0]

    M = pdist(x, y)  # distance_matrix(x,y,p=2)

    '''estimate lambda and delta'''
    M_mean = torch.mean(M)
    M_drop = F.dropout(M, 0.5 / (nx * ny))
    delta = torch.max(M_drop).detach()
    eff_lam = (lam / M_mean).detach()

    '''compute new distance matrix'''
    Mt = M
    row = delta * torch.ones(M[0:1, :].shape).to(device)
    col = torch.cat([delta * torch.ones(M[:, 0:1].shape).to(device), torch.zeros((1, 1)).to(device)], 0)
    if cuda:
        row = row.to(device)
        col = col.to(device)
    Mt = torch.cat([M, row], 0)
    Mt = torch.cat([Mt, col], 1)

    '''compute marginal'''
    a = torch.cat([p * torch.ones((nx, 1)) / nx, (1 - p) * torch.ones((1, 1))], 0)
    b = torch.cat([(1 - p) * torch.ones((ny, 1)) / ny, p * torch.ones((1, 1))], 0)

    '''compute kernel'''
    Mlam = eff_lam * Mt
    temp_term = torch.ones(1) * 1e-6
    if cuda:
        temp_term = temp_term.to(device)
        a = a.to(device)
        b = b.to(device)
    K = torch.exp(-Mlam) + temp_term
    U = K * Mt
    ainvK = K / a

    u = a

    for i in range(its):
        u = 1.0 / (ainvK.matmul(b / torch.t(torch.t(u).matmul(K))))
        if cuda:
            u = u.to(device)
    v = b / (torch.t(torch.t(u).matmul(K)))
    if cuda:
        v = v.to(device)

    upper_t = u * (torch.t(v) * K).detach()

    E = upper_t * Mt
    D = 2 * torch.sum(E)
    D /= scal

    if cuda:
        D = D.to(device)

    return D, Mlam

def continuous_to_categorical_probs(x):
    # 假设 x 是已经归一化到 [0, 1] 的连续值
    # 线性变换，将输入映射到三个 logits 上
    logits = torch.tensor([1.0, 2.0, 3.0]) * x  # 这里你可以根据需求调整逻辑系数
    # 通过 softmax 将 logits 转换为概率分布
    probs = F.softmax(logits, dim=0)
    return probs