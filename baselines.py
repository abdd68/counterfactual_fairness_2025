from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn import svm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import numpy as np
import torch
from utils import *
from models import CausalModel_law_up, CausalModel_law, CausalModel_adult, CausalModel_adult_up, CausalModel_crime_up
import pyro
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from main import args, logger
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# constant predictor
def run_constant(x, y, train_idx, test_idx, type='linear'):
    y_const = np.mean(y[train_idx], axis=0)  # the average y in train data
    if type == 'logistic':
        y_const = 1.0 if y_const >= 0.5 else 0.0
    y_pred_tst = np.full(len(test_idx), y_const)
    return y_pred_tst

# full predictor
def run_full(x, y, env, train_idx, test_idx, type='linear'):
    if type == 'linear':
        model = LinearRegression()  # linear ? logistic ?
    else:
        model = LogisticRegression(class_weight='balanced')

    features_full = np.concatenate([x, env], axis=1)
    features_full_trn = features_full[train_idx]
    features_full_tst = features_full[test_idx]

    model.fit(features_full_trn, y[train_idx])  # train

    # test
    y_pred_tst = model.predict(features_full_tst)
    return y_pred_tst, model

# unaware predictor
def run_unaware(x, y, train_idx, test_idx, type='linear'):
    if type == 'linear':
        model = LinearRegression()
    else:
        model = LogisticRegression(class_weight='balanced')

    model.fit(x[train_idx], y[train_idx])  # train
    # test
    y_pred_tst = model.predict(x[test_idx])

    return y_pred_tst, model

def run_fairk(data_save, train_idx, test_idx, type='linear', device = 'cpu'):
    if args.dataset == 'law':
        model = CausalModel_law(args, 'fairk_law').to(device)
        data_save = to_tensor(data_save)
        model, guide = train_casual(model, data_save, train_idx)
        data_return = guide()
        x_fair = data_return['knowledge'].view(-1, 1).cpu().detach().numpy()
        data_y = data_save['data']['ZFYA'].cpu().detach().numpy()
        x_fair_train, x_fair_test, data_y_train = x_fair[train_idx], x_fair[test_idx], data_y[train_idx]
        
    elif args.dataset == 'adult':
        model = CausalModel_adult(args, 'fairk_adult').to(device)
        data_save = to_tensor(data_save)
        model, guide = train_casual(model, data_save, train_idx)
        data_return = guide()
        latent_var = torch.cat([data_return['eps_MaritalStatus'].view(-1, 1),
                            data_return['eps_Occupation'].view(-1, 1), data_return['eps_EducationNum'].view(-1, 1),
                            data_return['eps_HoursPerWeek'].view(-1, 1), data_return['eps_Income'].view(-1, 1)],
                           dim=1).cpu().detach().numpy()
        x_fair = np.concatenate([data_save['data']['Sex'].cpu().detach().numpy(), latent_var], axis=1)
        data_y = data_save['data']['Income'].cpu().detach().numpy()
        x_fair_train, x_fair_test, data_y_train = x_fair[train_idx], x_fair[test_idx], data_y[train_idx]
        
    if args.dataset == 'law':
        model_linear = LinearRegression()
    elif args.dataset == 'adult':
        model_linear = LogisticRegression(class_weight='balanced')
        
    model_linear.fit(x_fair_train, data_y_train.reshape(-1))  # train
    y_pred_test = model_linear.predict(x_fair_test)  # n_tst
        
    return y_pred_test, model_linear, data_return

def run_exoc(data_save, train_idx, test_idx, type='linear', device = 'cpu'):
    if args.dataset == 'law':
        model = CausalModel_law_up(args, 'exoc').to(device)
    elif args.dataset == 'adult':
        model = CausalModel_adult_up(args, 'exoc').to(device)
    elif args.dataset == 'crimes':
        model = CausalModel_crime_up(args, 'exoc').to(device)
    # data_save = to_tensor(data_save)
    model, guide = train_casual_up(model, data_save, train_idx)
    
    data_return = guide()
    if args.dataset == 'law':
        x_fair = data_return['knowledge'].view(-1, 1).cpu().detach().numpy()
        data_y = data_save['data']['ZFYA'].cpu().detach().numpy()
    elif args.dataset == 'adult':
        latent_var = torch.cat([data_return['eps_MaritalStatus'].view(-1, 1),
                            data_return['eps_Occupation'].view(-1, 1), data_return['eps_EducationNum'].view(-1, 1),
                            data_return['eps_HoursPerWeek'].view(-1, 1), data_return['eps_Income'].view(-1, 1)],
                           dim=1).cpu().detach().numpy()
        x_fair = np.concatenate([data_save['data']['Sex'].cpu().detach().numpy(), latent_var], axis=1)
        data_y = data_save['data']['Income'].cpu().detach().numpy()
    elif args.dataset == 'crimes':
        x_fair = data_return['knowledge'].view(-1, 1).cpu().detach().numpy()
        data_y = data_save['data']['label'].cpu().detach().numpy()
        
    x_fair_train, x_fair_test, data_y_train = x_fair[train_idx], x_fair[test_idx], data_y[train_idx]
    
    if args.dataset == 'law' or args.dataset == 'crimes':
        model_linear = LinearRegression()
    elif args.dataset == 'adult':
        model_linear = LogisticRegression(class_weight='balanced')
        
    model_linear.fit(x_fair_train, data_y_train.reshape(-1))  # train
    y_pred_test = model_linear.predict(x_fair_test)  # n_tst
        
    return y_pred_test, model_linear, data_return

def train_casual(model, data_save, train_idx):
    def get_train_set(data_save):
        train_set = {}
        for key in data_save['data']:
            train_set[key] = data_save['data'][key][train_idx,:]
        return train_set

    def get_dataset(data_save):
        train_set = {}
        for key in data_save['data']:
            train_set[key] = data_save['data'][key]
        return train_set
    
    num_train = len(train_idx)
    if args.training:  # train
        guide = AutoDiagonalNormal(model)
        adam = pyro.optim.Adam({"lr": args.lrcausalmodel})
        svi = SVI(model, guide, adam, loss=Trace_ELBO())
        pyro.clear_param_store()
        # train model
        for j in range(args.num_iterations_causalmodel):
            # calculate the loss and take a gradient step
            loss = svi.step(get_dataset(data_save))  # all data is used here # dict_keys(['LSAT', 'UGPA', 'ZFYA', 'race', 'sex'])
            
        # save
        save_flag = True
        if save_flag:
            torch.save({"model": model.state_dict(), "guide": guide}, args.pyromodel)
            pyro.get_param_store().save(args.pyroparam)
            logger.info(f'saved causal model in: {args.pyromodel}')
    else:  # load
        saved_model_dict = torch.load(args.pyromodel)
        model.load_state_dict(saved_model_dict['model'])
        guide = saved_model_dict['guide']
        pyro.get_param_store().load(args.pyroparam)

        logger.info(f'loaded causal model from: {args.pyromodel}')
        
    return model, guide

cc = 0
def train_casual_up(model, data_save, train_idx):
    def get_dataset(data_save):
        train_set = {}
        for key in data_save['data']:
            train_set[key] = data_save['data'][key]
        return train_set
    
    def custom_loss(a, b):
        loss = F.mse_loss(a, b)
        return loss
    def elbo_loss(model, guide, *_args, **kwargs):
        global cc
        guide_a, guide_b = pyro.poutine.trace(guide).get_trace(*_args, **kwargs).nodes["sp"]["value"], \
                           pyro.poutine.trace(guide).get_trace(*_args, **kwargs).nodes["spp"]["value"]
                           

        # # ELBO loss and custom loss
        elbo_loss = Trace_ELBO().differentiable_loss(model, guide, *_args, **kwargs)
        # Computes a dynamic constant such that the result of multiplying custom_loss by this constant is the same order of magnitude as elbo_loss
        scale_factor = elbo_loss.item() / (20 * custom_loss(guide_a, guide_b).item() + 1e-8)  # avoid dividing zero
        customloss = args.gamma * scale_factor * custom_loss(guide_a, guide_b)
        loss = elbo_loss + customloss

        if cc % 1000 == 0:
            logger.info(f"[iteration {cc}] elbo_loss:{elbo_loss:.2f}, custom_loss:{customloss:.2f}")
        cc += 1
        return loss
    
    num_train = len(train_idx)
    if args.training:  # train
        guide = AutoDiagonalNormal(model)
        adam = pyro.optim.Adam({"lr": args.lrcausalmodel})
        svi = SVI(model, guide, adam, loss=elbo_loss)
        pyro.clear_param_store()
        # train model
        for j in range(args.num_iterations_causalmodel_up):
            # calculate the loss and take a gradient step
            loss = svi.step(get_dataset(data_save))  # all data is used here # dict_keys(['LSAT', 'UGPA', 'ZFYA', 'race', 'sex'])
            if j % 10000 == 0:
                logger.info("[iteration %04d] loss: %.4f" % (j + 1, loss / num_train))
            
        # save
        save_flag = True
        if save_flag:
            torch.save({"model": model.state_dict(), "guide": guide}, args.pyromodel)
            pyro.get_param_store().save(args.pyroparam)
            logger.info(f'saved causal model in: {args.pyromodel}')
    else:  # load
        saved_model_dict = torch.load(args.pyromodel)
        model.load_state_dict(saved_model_dict['model'])
        guide = saved_model_dict['guide']
        pyro.get_param_store().load(args.pyroparam)

        logger.info(f'loaded causal model from: {args.pyromodel}')
        
    return model, guide

def to_tensor(data_save):
    data = data_save['data']
    index = data_save['index']
    for k in data:
        data[k] = torch.FloatTensor(data[k])
        if args.cuda:
            data[k] = data[k].to(args.device)
    for k in index:
        index[k] = torch.LongTensor(index[k])
        if args.cuda:
            index[k] = index[k].to(args.device)
    if args.dataset == 'synthetic':
        param = data_save['params']
        return {'data': data, 'params': param, 'index': index}
    return {'data': data, 'index': index}