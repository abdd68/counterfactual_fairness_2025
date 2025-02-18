import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist

from torch import nn
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.poutine.trace_messenger import TraceMessenger
import pyro.distributions.constraints as constraints


pyro.enable_validation(True)
pyro.set_rng_seed(1)
pyro.enable_validation(True)
INF_LOW = 1e-16

import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def quickprocess(var):
    if var is None:
        return var
    var = var.view(-1).to(device)
    return var

def to_onehot(var, num_classes=-1):
    var_onehot = F.one_hot(var, num_classes)
    dim = num_classes if num_classes != -1 else var_onehot.shape[1]
    return var_onehot, dim

def onehot_to_int(var):
    var_int = torch.argmax(var, dim=1)
    return var_int

class CausalModel_law(PyroModule):
    def __init__(self, args, model_name):
        super().__init__()
        global device
        device = args.device
        self.model_name = model_name
        self.one_hot = 0

    def forward(self, data):
        dim_race = 1
        data_race, data_UGPA, data_LSAT, data_ZFYA = data['race'], data['UGPA'], data['LSAT'], data['ZFYA']
        data_race, data_UGPA, data_LSAT, data_ZFYA = quickprocess(data_race), quickprocess(data_UGPA), quickprocess(data_LSAT), quickprocess(data_ZFYA)
        if data_LSAT is not None:
            data_LSAT = torch.floor(data_LSAT)
        if self.one_hot:
            dim_race = 3

        self.pi = pyro.param(self.model_name + "_" + "pi", torch.tensor([0.4, 0.3, 0.3])).to(device)  # S~Cate(pi)

        self.b_g = pyro.param(self.model_name + "_" + "b_g", torch.tensor(0.)).to(device)
        self.w_g_k = pyro.param(self.model_name + "_" + "w_g_k", torch.tensor(0.)).to(device)
        self.w_g_r = pyro.param(self.model_name + "_" + "w_g_r", torch.zeros(dim_race, 1)).to(device)
        self.sigma_g = pyro.param(self.model_name + "_" + "sigma_g", torch.tensor(1.)).to(device)

        self.b_l = pyro.param(self.model_name + "_" + "b_l", torch.tensor(0.)).to(device)
        self.w_l_k = pyro.param(self.model_name + "_" + "w_l_k", torch.tensor(0.)).to(device)
        self.w_l_r = pyro.param(self.model_name + "_" + "w_l_r", torch.zeros(dim_race, 1)).to(device)

        self.w_f_k = pyro.param(self.model_name + "_" + "w_f_k", torch.tensor(0.)).to(device)
        self.w_f_r = pyro.param(self.model_name + "_" + "w_f_r", torch.zeros(dim_race, 1)).to(device)

        n = len(data_race)
        with pyro.plate('observe_data', size=n, device=device):
            knowledge = pyro.sample('knowledge', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.)).to(device)  # prior, n
            race = pyro.sample('obs_race', pyro.distributions.Categorical(self.pi), obs=data_race)  # S ~ Categorical(pi)
            race_out = race
            if self.one_hot:
                race_out, dim_race = to_onehot(data_race.long(), 3)
                race_out = race_out.float()
            gpa_mean = self.b_g + self.w_g_k * knowledge + (race_out.view(-1,dim_race) @ self.w_g_r).view(-1)
            sat_mean = torch.exp(self.b_l + self.w_l_k * knowledge + (race_out.view(-1,dim_race) @ self.w_l_r).view(-1))
            fya_mean = self.w_f_k * knowledge + (race_out.view(-1,dim_race) @ self.w_f_r).view(-1)

            gpa_obs = pyro.sample("obs_UGPA", dist.Normal(gpa_mean, torch.abs(self.sigma_g)), obs=data_UGPA)
            sat_obs = pyro.sample("obs_LSAT", dist.Poisson(sat_mean), obs=data_LSAT)
            fya_obs = pyro.sample("obs_ZFYA", dist.Normal(fya_mean, 1), obs=data_ZFYA)

        data_return = {'knowledge': knowledge, 'LSAT': sat_obs, 'ZFYA': fya_obs, 'UGPA': gpa_obs, 'race': race}
        return data_return
    
class CausalModel_law_up(PyroModule):
    def __init__(self, args, model_name):
        super().__init__()
        global device 
        device = args.device
        self.model_name = model_name
        self.args = args

    def forward(self, data):
        data_race, data_UGPA, data_LSAT, data_ZFYA = data['race'], data['UGPA'], data['LSAT'], data['ZFYA']
        data_race, data_UGPA, data_LSAT, data_ZFYA = quickprocess(data_race), quickprocess(data_UGPA), quickprocess(data_LSAT), quickprocess(data_ZFYA)
        if data_LSAT is not None:
            data_LSAT = torch.floor(data_LSAT)

        self.b_sp = pyro.param(self.model_name + "_" + "b_sp", torch.tensor(0.)).to(device)
        self.w_sp_k = pyro.param(self.model_name + "_" + "w_sp_k", torch.tensor(0.)).to(device)
        self.sigma_sp = pyro.param(self.model_name + "_" + "sigma_sp", torch.tensor(1.)).to(device)

        self.b_r = pyro.param(self.model_name + "_" + "b_r", torch.tensor(0.)).to(device)
        self.w_r_sp = pyro.param(self.model_name + "_" + "w_r_sp", torch.tensor(0.)).to(device)
        self.sigma_r = pyro.param(self.model_name + "_" + "sigma_r", torch.tensor(1.)).to(device)

        self.b_g = pyro.param(self.model_name + "_" + "b_g", torch.tensor(0.)).to(device)
        self.w_g_k = pyro.param(self.model_name + "_" + "w_g_k", torch.tensor(0.)).to(device)
        self.w_g_sp = pyro.param(self.model_name + "_" + "w_g_sp", torch.tensor(0.)).to(device)
        self.w_g_s = pyro.param(self.model_name + "_" + "w_g_s", torch.tensor(0.)).to(device)
        self.sigma_g = pyro.param(self.model_name + "_" + "sigma_g", torch.tensor(1.)).to(device)

        self.b_l = pyro.param(self.model_name + "_" + "b_l", torch.tensor(0.)).to(device)
        self.w_l_k = pyro.param(self.model_name + "_" + "w_l_k", torch.tensor(0.)).to(device)
        self.w_l_s = pyro.param(self.model_name + "_" + "w_l_s", torch.tensor(0.)).to(device)
        self.w_l_sp = pyro.param(self.model_name + "_" + "w_l_sp", torch.tensor(0.)).to(device)

        self.w_f_k = pyro.param(self.model_name + "_" + "w_f_k", torch.tensor(0.)).to(device)
        self.w_f_sp = pyro.param(self.model_name + "_" + "w_f_sp", torch.tensor(0.)).to(device)
        
        self.b_spp = pyro.param(self.model_name + "_" + "b_spp", torch.tensor(0.)).to(device)
        self.w_spp_f = pyro.param(self.model_name + "_" + "w_spp_f", torch.tensor(0.)).to(device)

        n = len(data_race)
        with pyro.plate('observe_data', size=n, device=device):
            knowledge = pyro.sample('knowledge', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.)).to(device)  # U ->
            sp_mean = self.b_sp + self.w_sp_k * knowledge
            sp = pyro.sample('sp', pyro.distributions.Normal(sp_mean, torch.abs(self.sigma_sp))).to(device) # S' <- U
            
            race_mean = self.b_r + self.w_r_sp * sp
            race = pyro.sample('obs_race', pyro.distributions.Normal(race_mean, torch.abs(self.sigma_r)), obs=data_race)  # S <- S'
            
            gpa_mean = self.b_g + self.w_g_k * knowledge + (sp * self.w_g_sp).view(-1) + (race * self.w_g_s).view(-1)
            sat_mean = torch.exp(self.b_l + self.w_l_k * knowledge + (sp * self.w_l_sp).view(-1) + (race * self.w_l_s).view(-1))
            fya_mean = self.w_f_k * knowledge + (sp * self.w_f_sp).view(-1)

            gpa_obs = pyro.sample("obs_UGPA", dist.Normal(gpa_mean, torch.abs(self.sigma_g)), obs=data_UGPA) # G <- U & <- S'
            sat_obs = pyro.sample("obs_LSAT", dist.Poisson(sat_mean), obs=data_LSAT) # L <- U & <- S'
            fya_obs = pyro.sample("obs_ZFYA", dist.Normal(fya_mean, 1), obs=data_ZFYA) # Y <- U & <- S'

            if self.args.use_spp:
                spp_mean = self.b_spp + self.w_spp_f * fya_obs
            else:
                spp_mean = fya_obs
            spp = pyro.sample("spp", dist.Normal(spp_mean, 1)) # S" <- Y

        data_return = {'knowledge': knowledge, 'LSAT': sat_obs, 'ZFYA': fya_obs, 'UGPA': gpa_obs, 'race': race, 'sp': sp, 'spp': spp}
        return data_return

def argmax_withNan(x, dim=1):
    return None if x is None else torch.argmax(x, dim=dim)

class CausalModel_adult(PyroModule):
    def __init__(self, args, model_name):
        super().__init__()
        self.model_name = model_name
        global device
        device = args.device

    def forward(self, data):
        data_Race, data_Sex, data_MaritalStatus, data_Occupation, data_EducationNum, data_HoursPerWeek, data_Income = data['Race'], \
                 data['Sex'], data['MaritalStatus'], data['Occupation'], data['EducationNum'], data['HoursPerWeek'], data['Income']
        data_Race = data_Race.view(-1, 1)

        self.pi_Race = pyro.param(self.model_name + "_" + "pi", torch.tensor([0.4, 0.3, 0.3])).to(device)
        self.pi_Sex = pyro.param(self.model_name + "_" + "pi", torch.tensor([0.5, 0.5])).to(device)

        # marital status: ~ categorical, logits = wx + b
        m_size = 7 if data_MaritalStatus is None else data_MaritalStatus.shape[1]
        self.w_mar_race = pyro.param("w_mar_race", torch.zeros(data_Race.shape[1], m_size)).to(device)  # d x d'
        self.w_mar_sex = pyro.param("w_mar_sex", torch.zeros(data_Sex.shape[1], m_size)).to(device)

        # education: Normal (better after standardization) ~ N(mean, 1), mean = wx + eps_edu
        e_size = 1 if data_EducationNum is None else data_EducationNum.shape[1]
        self.w_edu_race = pyro.param("w_edu_race", torch.zeros(data_Race.shape[1], 1)).to(device)  # d x 1
        self.w_edu_sex = pyro.param("w_edu_sex", torch.zeros(data_Sex.shape[1], 1)).to(device)  # d x 1

        # hour per week
        h_size = 1 if data_HoursPerWeek is None else data_HoursPerWeek.shape[1]
        self.w_hour_race = pyro.param("w_hour_race", torch.zeros(data_Race.shape[1], 1)).to(device)  # d x 1
        self.w_hour_sex = pyro.param("w_hour_sex", torch.zeros(data_Sex.shape[1], 1)).to(device)
        self.w_hour_mar = pyro.param("w_hour_mar", torch.zeros(m_size, 1)).to(device)
        self.w_hour_edu = pyro.param("w_hour_edu", torch.zeros(e_size, 1)).to(device)

        # occupation
        o_size = 14 if data_Occupation is None else data_Occupation.shape[1]
        self.w_occ_race = pyro.param("w_occ_race", torch.zeros(data_Race.shape[1], o_size)).to(device)  # d x d'
        self.w_occ_sex = pyro.param("w_occ_sex", torch.zeros(data_Sex.shape[1], o_size)).to(device)
        self.w_occ_mar = pyro.param("w_occ_mar", torch.zeros(m_size, o_size)).to(device)
        self.w_occ_edu = pyro.param("w_occ_edu", torch.zeros(e_size, o_size)).to(device)

        # income
        self.w_income_race = pyro.param("w_income_race", torch.zeros(data_Race.shape[1], 2)).to(device)  # d x 2
        self.w_income_sex = pyro.param("w_income_sex", torch.zeros(data_Sex.shape[1], 2)).to(device)
        self.w_income_mar = pyro.param("w_income_mar", torch.zeros(m_size, 2)).to(device)
        self.w_income_edu = pyro.param("w_income_edu", torch.zeros(e_size, 2)).to(device)
        self.w_income_hour = pyro.param("w_income_hour", torch.zeros(h_size, 2)).to(device)
        self.w_income_occ = pyro.param("w_income_occ", torch.zeros(o_size, 2)).to(device)

        n = len(data_Race)

        with pyro.plate('observe_data', size=n, device=device):
            Race = pyro.sample('obs_Race', pyro.distributions.Categorical(self.pi_Race), obs=data_Race.view(-1)).view(-1, 1) # S ~ Categorical(pi)
            Sex = pyro.sample('obs_Sex', pyro.distributions.Categorical(self.pi_Sex), obs=torch.argmax(data_Sex, dim=1)).to(device)  # n, raw data
            Sex = F.one_hot(Sex, num_classes=data_Sex.shape[1]).float()  # raw -> one hot, n x d

            eps_MaritalStatus = pyro.sample('eps_MaritalStatus', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.)).view(-1, 1)
            MaritalStatus_logit = torch.softmax((torch.tile(eps_MaritalStatus, (1, m_size)) + torch.matmul(Race, self.w_mar_race) +
                                                torch.matmul(Sex, self.w_mar_sex)), dim=1)  # n x d_mar
            MaritalStatus = pyro.sample('obs_MaritalStatus', pyro.distributions.Categorical(MaritalStatus_logit),
                                        obs=argmax_withNan(data_MaritalStatus, dim=1)).to(device)   # n
            MaritalStatus = F.one_hot(MaritalStatus, num_classes=m_size).float()  # n x d

            eps_EducationNum = pyro.sample('eps_EducationNum', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.)).view(-1, 1)
            EducationNum_mean = torch.matmul(Race, self.w_edu_race) + torch.matmul(Sex, self.w_edu_sex) + eps_EducationNum  # n x 1
            EducationNum = pyro.sample("obs_EducationNum", pyro.distributions.Normal(EducationNum_mean.view(-1), 1.0), obs=quickprocess(data_EducationNum)).view(-1,1)  # n x 1

            eps_HoursPerWeek = pyro.sample('eps_HoursPerWeek', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.)).view(-1, 1)
            HoursPerWeek_mean = torch.matmul(Race, self.w_hour_race) + torch.matmul(Sex, self.w_hour_sex) + \
                                torch.matmul(MaritalStatus, self.w_hour_mar) + torch.matmul(EducationNum, self.w_hour_edu) + eps_HoursPerWeek
            HoursPerWeek = pyro.sample("obs_HoursPerWeek", pyro.distributions.Normal(HoursPerWeek_mean.view(-1), 1.0), obs=quickprocess(data_HoursPerWeek)).view(-1,1)  # n x 1

            eps_Occupation = pyro.sample('eps_Occupation', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.)).view(-1, 1)
            Occupation_logit = torch.softmax((torch.tile(eps_Occupation, (1, o_size)) +
                                              torch.matmul(Race, self.w_occ_race) + torch.matmul(Sex, self.w_occ_sex) +
                                              torch.matmul(EducationNum, self.w_occ_edu) + torch.matmul(MaritalStatus, self.w_occ_mar)), dim=1)  # n x d
            Occupation = pyro.sample('obs_Occupation', pyro.distributions.Categorical(Occupation_logit),
                                        obs=argmax_withNan(data_Occupation, dim=1)).to(device)  # n x 1
            Occupation = F.one_hot(Occupation, num_classes=o_size).float()  # n x d

            eps_Income = pyro.sample('eps_Income', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.)).view(-1, 1)  # n x 1
            Income_logit = torch.softmax((torch.tile(eps_Income, (1, 2)) + torch.matmul(Race, self.w_income_race) + torch.matmul(Sex, self.w_income_sex) +
                                         torch.matmul(MaritalStatus, self.w_income_mar) + torch.matmul(EducationNum, self.w_income_edu) +
                                         torch.matmul(HoursPerWeek, self.w_income_hour) + torch.matmul(Occupation, self.w_income_occ)), dim=1)  # n x 2
            Income = pyro.sample('obs_Income', pyro.distributions.Categorical(Income_logit),
                                        obs=argmax_withNan(data_Income, dim=1)).to(device).view(-1, 1).float()  # n x 1

        data_return = {'eps_MaritalStatus': eps_MaritalStatus, 'eps_EducationNum': eps_EducationNum,
                       'eps_Occupation': eps_Occupation, 'eps_HoursPerWeek': eps_HoursPerWeek,
                       'eps_Income': eps_Income, 'MaritalStatus': MaritalStatus, 'Occupation': Occupation,
                       'EducationNum': EducationNum, 'HoursPerWeek': HoursPerWeek, 'Income': Income,
                       'Race': Race, 'Sex': Sex
                       }
        return data_return

class CausalModel_adult_up(PyroModule):
    def __init__(self, args, model_name):
        super().__init__()
        self.model_name = model_name
        global device
        device = args.device
        self.args = args

    def forward(self, data):
        data_len = data['Race'].shape[0]
        data_Race, data_Sex, data_MaritalStatus, data_Occupation, data_EducationNum, data_HoursPerWeek, data_Income = data['Race'], \
                 data['Sex'], data['MaritalStatus'], data['Occupation'], data['EducationNum'], data['HoursPerWeek'], data['Income']
        data_Race = data_Race.view(-1, 1)

        self.pi_Race = pyro.param(self.model_name + "_" + "pi", torch.tensor([0.4, 0.3, 0.3]), constraint=constraints.positive).to(device)
        self.pi_Sex = pyro.param(self.model_name + "_" + "pi", torch.tensor([0.5, 0.5]), constraint=constraints.positive).to(device)

        # marital status: ~ categorical, logits = wx + b
        m_size = 7 if data_MaritalStatus is None else data_MaritalStatus.shape[1]
        self.w_mar_race = pyro.param("w_mar_race", torch.zeros(data_Race.shape[1], m_size)).to(device)  # d x d'

        self.w_mar_sex = pyro.param("w_mar_sex", torch.zeros(data_Sex.shape[1], m_size)).to(device)

        # education: Normal (better after standardization) ~ N(mean, 1), mean = wx + eps_edu
        e_size = 1 if data_EducationNum is None else data_EducationNum.shape[1]
        self.w_edu_race = pyro.param("w_edu_race", torch.zeros(data_Race.shape[1], 1)).to(device)  # d x 1
        self.w_edu_sex = pyro.param("w_edu_sex", torch.zeros(data_Sex.shape[1], 1)).to(device)  # d x 1

        # hour per week
        h_size = 1 if data_HoursPerWeek is None else data_HoursPerWeek.shape[1]
        self.w_hour_race = pyro.param("w_hour_race", torch.zeros(data_Race.shape[1], 1)).to(device)  # d x 1
        self.w_hour_sex = pyro.param("w_hour_sex", torch.zeros(data_Sex.shape[1], 1)).to(device)
        self.w_hour_mar = pyro.param("w_hour_mar", torch.zeros(m_size, 1)).to(device)
        self.w_hour_edu = pyro.param("w_hour_edu", torch.zeros(e_size, 1)).to(device)

        # occupation
        o_size = 14 if data_Occupation is None else data_Occupation.shape[1]
        self.w_occ_race = pyro.param("w_occ_race", torch.zeros(data_Race.shape[1], o_size)).to(device)  # d x d'
        self.w_occ_sex = pyro.param("w_occ_sex", torch.zeros(data_Sex.shape[1], o_size)).to(device)
        self.w_occ_mar = pyro.param("w_occ_mar", torch.zeros(m_size, o_size)).to(device)
        self.w_occ_edu = pyro.param("w_occ_edu", torch.zeros(e_size, o_size)).to(device)

        # income
        self.w_income_race = pyro.param("w_income_race", torch.zeros(data_Race.shape[1], 2)).to(device)  # d x 2
        self.w_income_sex = pyro.param("w_income_sex", torch.zeros(data_Sex.shape[1], 2)).to(device)
        self.w_income_mar = pyro.param("w_income_mar", torch.zeros(m_size, 2)).to(device)
        self.w_income_edu = pyro.param("w_income_edu", torch.zeros(e_size, 2)).to(device)
        self.w_income_hour = pyro.param("w_income_hour", torch.zeros(h_size, 2)).to(device)
        self.w_income_occ = pyro.param("w_income_occ", torch.zeros(o_size, 2)).to(device)
        
        self.eps_sp = pyro.param(self.model_name + "_" + "eps_sp", torch.tensor(0.)).to(device)
        self.w_sp_eps1 = pyro.param(self.model_name + "_" + "w_sp_eps1", torch.zeros(1, 1)).to(device)
        self.w_sp_eps2 = pyro.param(self.model_name + "_" + "w_sp_eps2", torch.zeros(1, 1)).to(device)
        self.w_sp_eps3 = pyro.param(self.model_name + "_" + "w_sp_eps3", torch.zeros(1, 1)).to(device)
        self.w_sp_eps4 = pyro.param(self.model_name + "_" + "w_sp_eps4", torch.zeros(1, 1)).to(device)
        self.w_sp_eps5 = pyro.param(self.model_name + "_" + "w_sp_eps5", torch.zeros(1, 1)).to(device)
        
        self.w_race_sp = pyro.param(self.model_name + "_" + "w_race_sp", torch.zeros(1, 1)).to(device)
        self.eps_race = pyro.param(self.model_name + "_" + "eps_race", torch.tensor(0.)).to(device)
        self.w_mar_sp = pyro.param(self.model_name + "_" + "w_mar_sp", torch.zeros(1, m_size)).to(device)
        self.w_occ_sp = pyro.param(self.model_name + "_" + "w_occ_sp", torch.zeros(1, o_size)).to(device)
        self.w_edu_sp = pyro.param(self.model_name + "_" + "w_edu_sp", torch.zeros(1, e_size)).to(device)
        self.w_hour_sp = pyro.param(self.model_name + "_" + "w_hour_sp", torch.zeros(1, h_size)).to(device)
        self.w_income_sp = pyro.param(self.model_name + "_" + "w_income_sp", torch.zeros(1, 2)).to(device)
        
        self.b_spp = pyro.param(self.model_name + "_" + "b_spp", torch.tensor(0.)).to(device)
        self.w_spp_f = pyro.param(self.model_name + "_" + "w_spp_f", torch.tensor(0.)).to(device)

        n = len(data_Race)

        with pyro.plate('observe_data', size=n, device=device):
            eps_MaritalStatus = pyro.sample('eps_MaritalStatus', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.)).view(-1, 1)
            eps_EducationNum = pyro.sample('eps_EducationNum', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.)).view(-1, 1)
            eps_HoursPerWeek = pyro.sample('eps_HoursPerWeek', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.)).view(-1, 1)
            eps_Occupation = pyro.sample('eps_Occupation', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.)).view(-1, 1)
            eps_Income = pyro.sample('eps_Income', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.)).view(-1, 1)  # n x 1
            sp_mean = self.eps_sp + torch.matmul(eps_MaritalStatus, self.w_sp_eps1) + torch.matmul(eps_EducationNum, self.w_sp_eps2) \
                + torch.matmul(eps_HoursPerWeek, self.w_sp_eps3) + torch.matmul(eps_Occupation, self.w_sp_eps4) + torch.matmul(eps_Income, self.w_sp_eps5)
            sp = pyro.sample('sp', pyro.distributions.Normal(sp_mean.view(-1).to(device), 1.)).view(-1, 1)  # n x 1
            
            race_mean = self.eps_race + torch.matmul(sp, self.w_race_sp)
            Race = pyro.sample('obs_Race', pyro.distributions.Normal(race_mean.view(-1), 1.), obs=data_Race.view(-1)).view(data_len, -1)  # S <- S'
            Sex = pyro.sample('obs_Sex', pyro.distributions.Categorical(self.pi_Sex), obs=torch.argmax(data_Sex, dim=1)).to(device)  # n, raw data
            Sex = F.one_hot(Sex, num_classes=data_Sex.shape[1]).float()  # raw -> one hot, n x d
            MaritalStatus_logit = torch.softmax((torch.tile(eps_MaritalStatus, (1, m_size)) + torch.matmul(Race, self.w_mar_race) +
                                                torch.matmul(Sex, self.w_mar_sex) + torch.matmul(sp, self.w_mar_sp)), dim=1)  # n x d_mar
            MaritalStatus = pyro.sample('obs_MaritalStatus', pyro.distributions.Categorical(MaritalStatus_logit),
                                        obs=argmax_withNan(data_MaritalStatus, dim=1)).to(device)   # n
            MaritalStatus = F.one_hot(MaritalStatus, num_classes=m_size).float()  # n x d

            
            EducationNum_mean = torch.matmul(Race, self.w_edu_race) + torch.matmul(Sex, self.w_edu_sex) + torch.matmul(sp, self.w_edu_sp) + eps_EducationNum  # n x 1
            EducationNum = pyro.sample("obs_EducationNum", pyro.distributions.Normal(EducationNum_mean.view(-1), 1.0), obs=quickprocess(data_EducationNum)).view(-1,1)  # n x 1

            
            HoursPerWeek_mean = torch.matmul(Race, self.w_hour_race) + torch.matmul(Sex, self.w_hour_sex) + \
                                torch.matmul(MaritalStatus, self.w_hour_mar) + torch.matmul(EducationNum, self.w_hour_edu) + torch.matmul(sp, self.w_hour_sp) + eps_HoursPerWeek
            HoursPerWeek = pyro.sample("obs_HoursPerWeek", pyro.distributions.Normal(HoursPerWeek_mean.view(-1), 1.0), obs=quickprocess(data_HoursPerWeek)).view(-1,1)  # n x 1

            Occupation_logit = torch.softmax((torch.tile(eps_Occupation, (1, o_size)) +
                                              torch.matmul(Race, self.w_occ_race) + torch.matmul(Sex, self.w_occ_sex) +
                                              torch.matmul(EducationNum, self.w_occ_edu) + torch.matmul(MaritalStatus, self.w_occ_mar) + torch.matmul(sp, self.w_occ_sp)), dim=1)  # n x d
            Occupation = pyro.sample('obs_Occupation', pyro.distributions.Categorical(Occupation_logit),
                                        obs=argmax_withNan(data_Occupation, dim=1)).to(device)  # n x 1
            Occupation = F.one_hot(Occupation, num_classes=o_size).float()  # n x d

            Income_logit = torch.softmax((torch.tile(eps_Income, (1, 2)) + torch.matmul(Race, self.w_income_race) + torch.matmul(Sex, self.w_income_sex) +
                                         torch.matmul(MaritalStatus, self.w_income_mar) + torch.matmul(EducationNum, self.w_income_edu) +
                                         torch.matmul(HoursPerWeek, self.w_income_hour) + torch.matmul(Occupation, self.w_income_occ) + torch.matmul(sp, self.w_income_sp)), dim=1)  # n x 2
            Income = pyro.sample('obs_Income', pyro.distributions.Categorical(Income_logit),
                                        obs=argmax_withNan(data_Income, dim=1)).to(device).view(-1, 1).float()  # n x 1
            if self.args.use_spp:
                spp_mean = self.b_spp + self.w_spp_f * Income
            else:
                spp_mean = Income
            spp = pyro.sample("spp", dist.Normal(spp_mean.view(-1), 1)).view(data_len, -1) # S" <- Y

        data_return = {'eps_MaritalStatus': eps_MaritalStatus, 'eps_EducationNum': eps_EducationNum,
                       'eps_Occupation': eps_Occupation, 'eps_HoursPerWeek': eps_HoursPerWeek,
                       'eps_Income': eps_Income, 'MaritalStatus': MaritalStatus, 'Occupation': Occupation,
                       'EducationNum': EducationNum, 'HoursPerWeek': HoursPerWeek, 'Income': Income,
                       'Race': Race, 'Sex': Sex, 'sp': sp, 'spp': spp
                       }
        return data_return
    
class Causal_model_vae(nn.Module):
    def __init__(self, dim_x, num_s, dim_h):
        super(Causal_model_vae, self).__init__()
        from main import args
        device = args.device
        self.device = args.device
        self.decoder_type = args.decoder_type
        self.num_s = num_s
        self.dim_x = dim_x
        self.dim_h = dim_h

        self.mu_h = nn.Sequential(nn.Linear(dim_x, self.dim_h), nn.LeakyReLU(), nn.Linear(self.dim_h, self.dim_h)).to(device)
        self.logvar_h = nn.Sequential(nn.Linear(dim_x, self.dim_h), nn.LeakyReLU(), nn.Linear(self.dim_h, self.dim_h)).to(device)

        if self.decoder_type == 'together':  # train a decoder: H + S -> X (non-sensitive features)
            self.decoder_elem = nn.Sequential(nn.Linear(self.dim_h + 1, self.dim_h), nn.LeakyReLU(), nn.Linear(self.dim_h, dim_x)).to(device)
        elif self.decoder_type == 'separate':  # separately train a decoder for each S: H -> X
            self.decoder_elem = [nn.Sequential(nn.Linear(self.dim_h, dim_h), nn.LeakyReLU(), nn.Linear(self.dim_h, dim_x)).to(device) for i in range(num_s)]

    def encoder(self, data):
        mu_h = self.mu_h(data)  # x, y
        logvar_h = self.logvar_h(data)

        return mu_h, logvar_h

    def get_embeddings(self, data):
        mu_h, logvar_h = self.encoder(data)
        return mu_h, logvar_h

    def reparameterize(self, mu, logvar):
        if self.training:
            # do this only while training
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decoder(self, h_sample, s):
        from main import args
        if self.decoder_type == 'together':  #
            input_dec = torch.cat([h_sample, s], dim=1)
            data_reconst = self.decoder_elem(input_dec)
        elif self.decoder_type == 'separate':
            data_reconst = torch.zeros((len(h_sample), self.dim_x+1)).to(args.device)
            for i in range(self.num_s):
                idx_si = torch.where(s == i)
                input_dec_i = h_sample[idx_si]
                data_reconst_i = self.decoder_elem[i](input_dec_i)
                data_reconst[idx_si] = data_reconst_i
        return data_reconst

    def forward(self, data, s):  # data: n x d, s: n x 1
        mu_h, logvar_h = self.encoder(data)
        h_sample = self.reparameterize(mu_h, logvar_h)

        data_reconst = self.decoder(h_sample, s)

        result_all = {'reconstruct': data_reconst, 'mu_h': mu_h, 'logvar_h': logvar_h, 'h_sample': h_sample}
        return result_all

    def get_latent_var(self, data):
        mu_h, logvar_h = self.encoder(data)
        h_sample = self.reparameterize(mu_h, logvar_h)
        h_sample = h_sample.view(-1, self.dim_h).cpu().detach().numpy()
        return h_sample
    

class CausalModel_crime_up(PyroModule):
    def __init__(self, args, model_name):
        super().__init__()
        global device
        device = args.device
        self.model_name = model_name

    def forward(self, data):
        # Data preparation
        data_race = data['race']
        data_X1 = data['population']
        data_X2 = data['medIncome']
        data_X3 = data['PctPopUnderPov']
        data_X4 = data['PctUnemployed']
        data_X5 = data['PctNotHSGrad']
        data_Y = data['label']

        # Parameters for the causal model
        dim_race = 1

        # Race (S) distribution
        self.pi = pyro.param(self.model_name + "_pi", torch.tensor([0.4, 0.3, 0.3])).to(device)  # S ~ Cate(pi)

        # Parameters for S'
        self.b_sp = pyro.param(self.model_name + "_b_sp", torch.tensor(0.)).to(device)
        self.w_sp_s = pyro.param(self.model_name + "_w_sp_s", torch.zeros(dim_race, 1)).to(device)

        # Parameters for X1-X5
        self.b_x = pyro.param(self.model_name + "_b_x", torch.zeros(5)).to(device)
        self.w_x_sp = pyro.param(self.model_name + "_w_x_sp", torch.zeros(1, 5)).to(device)

        # Parameters for Y
        self.b_y = pyro.param(self.model_name + "_b_y", torch.tensor(0.)).to(device)
        self.w_y_sp = pyro.param(self.model_name + "_w_y_sp", torch.zeros(1)).to(device)
        self.w_y_x = pyro.param(self.model_name + "_w_y_x", torch.zeros(5, 1)).to(device)

        # Parameters for S''
        self.b_spp = pyro.param(self.model_name + "_b_spp", torch.tensor(0.)).to(device)
        self.w_spp_y = pyro.param(self.model_name + "_w_spp_y", torch.zeros(1)).to(device)

        n = len(data_race)
        with pyro.plate('observe_data', size=n, device=device):
            # S (race)
            race = pyro.sample('obs_race', dist.Categorical(self.pi), obs=data_race)

            # S' (sp)
            sp_mean = self.b_sp + (race.view(-1, dim_race) @ self.w_sp_s).view(-1)
            sp = pyro.sample('sp', dist.Normal(sp_mean, 1.))

            # X1-X5
            x_means = self.b_x + (sp.view(-1, 1) @ self.w_x_sp).view(-1, 5)

            # Ensure each observation is (n, 1) for concatenation
            x1_obs = pyro.sample("obs_X1", dist.Normal(x_means[:, 0], 1.), obs=data_X1).unsqueeze(-1)
            x2_obs = pyro.sample("obs_X2", dist.Normal(x_means[:, 1], 1.), obs=data_X2).unsqueeze(-1)
            x3_obs = pyro.sample("obs_X3", dist.Normal(x_means[:, 2], 1.), obs=data_X3).unsqueeze(-1)
            x4_obs = pyro.sample("obs_X4", dist.Normal(x_means[:, 3], 1.), obs=data_X4).unsqueeze(-1)
            x5_obs = pyro.sample("obs_X5", dist.Normal(x_means[:, 4], 1.), obs=data_X5).unsqueeze(-1)

            # Concatenate and compute Y
            y_mean = self.b_y + sp.view(-1) * self.w_y_sp + \
                     (torch.cat([x1_obs, x2_obs, x3_obs, x4_obs, x5_obs], dim=1) @ self.w_y_x).view(-1)
            y_obs = pyro.sample("obs_Y", dist.Normal(y_mean, 1.), obs=data_Y)

            # S'' (spp)
            spp_mean = self.b_spp + self.w_spp_y * y_obs
            spp = pyro.sample('spp', dist.Normal(spp_mean, 1.))

        # Return data
        data_return = {
            'sp': sp,
            'spp': spp,
            'X1': x1_obs,
            'X2': x2_obs,
            'X3': x3_obs,
            'X4': x4_obs,
            'X5': x5_obs,
            'Y': y_obs,
            'race': race
        }
        return data_return