import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
torch.set_default_tensor_type(torch.cuda.FloatTensor)

from typing import Tuple
import math
import numpy as np
import matplotlib.pyplot as plt
import gzip
import itertools
from scipy.spatial import KDTree    
import time
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing

# import copy

device = torch.device('cuda')

class StandardScaler(object):
    def __init__(self):
        pass

    def fit(self, data):

        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True) 
        self.std[self.std < 1e-12] = 1.0

        self.mu_tensor = torch.from_numpy(self.mu).float().to('cuda')
        self.std_tensor = torch.from_numpy(self.std).float().to('cuda')

    def transform(self, data):

        return (data - self.mu) / self.std

    def inverse_transform(self, data):

        return self.std * data + self.mu
    
    def transform_tensor(self, data):
        
        return (data - self.mu_tensor) / self.std_tensor

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x

def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)

class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0., bias: bool = True) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class EnsembleModel(nn.Module):
    # ensemble nn
    def __init__(self, state_size, action_size, reward_size, ensemble_size, hidden_size=200, learning_rate=1e-3, use_decay=False):
        super(EnsembleModel, self).__init__()
        self.hidden_size = hidden_size
        self.nn1 = EnsembleFC(state_size + action_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.use_decay = use_decay


        self.output_dim = state_size + reward_size
        self.reward_size = reward_size


        self.nn5 = EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)

        
        self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim)).float() / 2).to(device), requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim)).float() * 10).to(device), requires_grad=False)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.apply(init_weights)
        self.swish = Swish()

    def forward(self, x, mode='rs', ret_log_var=False):
        nn1_output = self.swish(self.nn1(x))
        nn2_output = self.swish(self.nn2(nn1_output))
        nn3_output = self.swish(self.nn3(nn2_output))
        nn4_output = self.swish(self.nn4(nn3_output))
        nn5_output = self.nn5(nn4_output)

        mean = nn5_output[:, :, :self.output_dim]

        logvar = self.max_logvar - F.softplus(self.max_logvar - nn5_output[:, :, self.output_dim:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if mode=='rs':
            if ret_log_var:
                return mean, logvar
            else:
                return mean, torch.exp(logvar)
        elif mode=='s':
            if ret_log_var:
                return mean[:, :, self.reward_size:], logvar[:, :, self.reward_size:]
            else:
                return mean[:, :, self.reward_size:], torch.exp(logvar[:, :, self.reward_size:])
        elif mode=='r':
            if ret_log_var:
                return mean[:, :, :self.reward_size], logvar[:, :, :self.reward_size]
            else:
                return mean[:, :, :self.reward_size], torch.exp(logvar[:, :, :self.reward_size])


    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.

        return decay_loss

    def loss(self, mean, logvar, labels, inc_var_loss=True):
        """
        mean, logvar: Ensemble_size x N x dim
        labels: N x dim
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-logvar)
        if inc_var_loss:

            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss
    

    def train(self, loss, loss_regular,weight_grad_loss=1000):
        gamma = weight_grad_loss
        self.optimizer.zero_grad()

        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        loss += gamma * loss_regular

        if self.use_decay:
            loss += self.get_decay_loss()
        loss.backward()

        self.optimizer.step()


class EnsembleDynamicsModel():
    def __init__(self, network_size, elite_size, state_size, action_size, reward_size=1, hidden_size=200, use_decay=False):
        self.network_size = network_size
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.network_size = network_size
        self.elite_model_idxes = []

        self.elite_model_idxes_reward = []

        self.ensemble_model = EnsembleModel(state_size, action_size, reward_size, network_size, hidden_size, use_decay=use_decay)
        self.scaler = StandardScaler()
        self.state_size = state_size
        self.action_size = action_size
        self.tree = None 

    def function_grad(self, x):
        x = x.view(self.network_size, -1, self.state_size+self.action_size)
        state = x[:,:,:self.state_size]
        x = self.scaler.transform_tensor(x)
        y, _ = self.ensemble_model(x, mode='rs', ret_log_var=True)
        y[:,:,self.reward_size:] += state

        return y.view(-1, self.state_size+self.reward_size, self.state_size+self.reward_size)  

    def train(self, inputs, labels, state_regular, action_regular, next_state_regular, reward_regular, batch_size=256, weight_grad_loss=10, holdout_ratio=0., max_epochs_since_update=5, near_n=5):
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.network_size)}

        num_holdout = int(inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]
        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]

        inputs_regular = np.concatenate((state_regular, action_regular), axis=-1) 
        labels_regular = np.concatenate((reward_regular.reshape([len(reward_regular),1]), next_state_regular), axis=-1) 
        num_holdout_regular = int(inputs_regular.shape[0] * holdout_ratio)*0
        permutation_regular = np.random.permutation(inputs_regular.shape[0])
        inputs_regular, labels_regular = inputs_regular[permutation_regular], labels_regular[permutation_regular]
        train_inputs_regular, train_labels_regular = inputs_regular[num_holdout_regular:], labels_regular[num_holdout_regular:]

        tree = KDTree(inputs_regular)

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])        


        global tree_query
        def tree_query(train_inputs_regular_i):
            _, ind = tree.query(train_inputs_regular_i,k=near_n+1)
            ind = ind[1:]
            return ind            
        
        index_near_n_all = np.zeros([train_inputs_regular.shape[0],near_n,1],dtype='int')


        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cores)
        train_inputs_regular_list = [train_inputs_regular[i,:] for i in range(train_inputs_regular.shape[0])]
        index_near_n_all_list = pool.map(tree_query, train_inputs_regular_list)
        pool.close()
        index_near_n_all = np.array(index_near_n_all_list, dtype='int').reshape([train_inputs_regular.shape[0],near_n,1])

        index_near_n_all_array = np.array(index_near_n_all_list)
        train_input_regular_near_all = (labels_regular[index_near_n_all_array]- train_labels_regular.reshape([train_labels_regular.shape[0],1,-1]).repeat(near_n,axis=1)).copy()
        train_input_regular_near_all = train_input_regular_near_all.reshape([train_inputs_regular.shape[0], near_n, train_labels_regular.shape[1]])
        V_all = (inputs_regular[index_near_n_all_array]- train_inputs_regular.reshape([train_inputs_regular.shape[0],1,-1]).repeat(near_n,axis=1)).copy()
        V_all = V_all.reshape([train_inputs_regular.shape[0], near_n, train_inputs_regular.shape[1]])


        for epoch in itertools.count():

            train_idx = np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(self.network_size)]) # num model * len data
            train_idx_regular = np.vstack([np.random.permutation(train_inputs_regular.shape[0]) for _ in range(self.network_size)]) # num model * len data
            
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[:, start_pos: start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[idx]).float().to(device) # num_model * batch * dim in
                train_label = torch.from_numpy(train_labels[idx]).float().to(device)
                mean, logvar = self.ensemble_model(train_input, mode='rs', ret_log_var=True)
                loss, _ = self.ensemble_model.loss(mean, logvar, train_label)

                #### regular
                if start_pos % train_inputs_regular.shape[0]< (start_pos + batch_size) % train_inputs_regular.shape[0]:
                    idx_regular = train_idx_regular[:, start_pos % train_inputs_regular.shape[0]: (start_pos + batch_size) % train_inputs_regular.shape[0]]
                else:
                    idx_regular = train_idx_regular[:, 0: (start_pos + batch_size) % train_inputs_regular.shape[0]]
                train_input_regular = torch.from_numpy(train_inputs_regular[idx_regular]).float().to(device) # num_model * batch * dim in
                train_label_regular = torch.from_numpy(train_labels_regular[idx_regular]).float().to(device)
                train_input_regular_near = np.zeros([self.network_size, train_input_regular.shape[1], near_n, train_labels_regular.shape[1]]) # nmodel, batch, near_n, dim s + dim a
                
                index_list = np.zeros([self.network_size, train_input_regular.shape[1], near_n, 1])# num_model * batch * near_n*1
                # t0 = time.time() ####
                for i in range(self.network_size):
                    for j in range(train_input_regular.shape[1]):
                        index_near5 = index_near_n_all[idx_regular[i,j]].reshape(near_n)
                        index_list[i,j,:] = index_near5.reshape(near_n,1)
                        train_input_regular_near[i,j,:,:] = train_input_regular_near_all[idx_regular[i,j]]

                loss_grad = 0

                train_input_regular_near = torch.from_numpy(train_input_regular_near).to(device)

                train_regular_grad = get_batch_jacobian(self.function_grad, train_input_regular.view(train_input_regular.shape[1]*self.network_size,-1), self.state_size+self.reward_size)
                train_regular_grad = train_regular_grad.view(self.network_size, train_input_regular.shape[1], self.state_size+self.reward_size, self.state_size+self.action_size)

                V = np.zeros([self.network_size, train_input_regular.shape[1],near_n,self.state_size+self.action_size])
                for j in range(train_input_regular.shape[1]):                    
                    for i in range(self.network_size):
                        V[i,j,:,:]=V_all[idx_regular[i,j]]

                V = torch.from_numpy(V).to(device)
                V = V.view(self.network_size, train_input_regular.shape[1],near_n,1,self.state_size+self.action_size)

                train_regular_grad = train_regular_grad.view(self.network_size, train_input_regular.shape[1], 1, self.state_size+self.reward_size, self.state_size+self.action_size)
                train_regular_grad = train_regular_grad.repeat(1,1,near_n,1,1)
                regular = torch.sum( train_regular_grad * V ,-1).view(self.network_size, train_input_regular.shape[1], near_n, self.state_size+self.reward_size)
                loss_grad = torch.mean(torch.pow( regular - train_input_regular_near, 2))
                
                self.ensemble_model.train(loss, loss_grad, weight_grad_loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar = self.ensemble_model(holdout_inputs, mode='rs', ret_log_var=True)
                _, holdout_mse_losses = self.ensemble_model.loss(holdout_mean, holdout_logvar, holdout_labels, inc_var_loss=False)
                holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
                sorted_loss_idx = np.argsort(holdout_mse_losses)
                self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()
                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    break

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                updated = True

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def predict(self, inputs, batch_size=1024, factored=True):
        inputs = self.scaler.transform(inputs)
        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(inputs[i:min(i + batch_size, inputs.shape[0])]).float().to(device)
            b_mean, b_var = self.ensemble_model(input[None, :, :].repeat([self.network_size, 1, 1]), ret_log_var=False)
            ensemble_mean.append(b_mean.detach().cpu().numpy())
            ensemble_var.append(b_var.detach().cpu().numpy())
        ensemble_mean = np.hstack(ensemble_mean)
        ensemble_var = np.hstack(ensemble_var)

        if factored:
            return ensemble_mean, ensemble_var
        else:
            assert False, "Need to transform to numpy"
            mean = torch.mean(ensemble_mean, dim=0)
            var = torch.mean(ensemble_var, dim=0) + torch.mean(torch.square(ensemble_mean - mean[None, :, :]), dim=0)
            return mean, var

    def predict_tensor(self, inputs, batch_size=1024, factored=True):
        inputs = self.scaler.transform_tensor(inputs)
        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = inputs[i:min(i + batch_size, inputs.shape[0])]
            b_mean, b_var = self.ensemble_model(input[None, :, :].repeat([self.network_size, 1, 1]), ret_log_var=False)
            ensemble_mean.append(b_mean)
            ensemble_var.append(b_var)
        ensemble_mean = torch.cat(ensemble_mean,1)  ##
        ensemble_var = torch.cat(ensemble_var,1)  ##

        if factored:
            return ensemble_mean, ensemble_var
        else:
            assert False, "Need to transform to numpy"
            mean = torch.mean(ensemble_mean, dim=0)
            var = torch.mean(ensemble_var, dim=0) + torch.mean(torch.square(ensemble_mean - mean[None, :, :]), dim=0)
            return mean, var


class EnsembleEnv():
    def __init__(self, network_size, elite_size, state_size, action_size, reward_size=1, hidden_size=200, use_decay=False):
        self.model = EnsembleDynamicsModel(network_size, elite_size, state_size, action_size, reward_size=1, hidden_size=200, use_decay=use_decay)

    def step(self, obs, act, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        ensemble_model_means[:, :, 1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape

        batch_idxes = np.arange(0, batch_size)
        
        model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)

        samples = ensemble_samples[model_idxes, batch_idxes]
        model_means = ensemble_model_means[model_idxes, batch_idxes]
        model_stds = ensemble_model_stds[model_idxes, batch_idxes]

        rewards, next_obs = samples[:, :1], samples[:, 1:]

        if return_single:
            next_obs = next_obs[0]
            rewards = rewards[0]

        return next_obs, rewards
    
    def step_tensor(self, obs, act, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = torch.cat((obs,act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict_tensor(inputs)
        ensemble_model_means[:, :, 1:] += obs
        ensemble_model_stds = torch.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + torch.randn(size=ensemble_model_means.shape) * ensemble_model_stds
                
        num_models, batch_size, _ = ensemble_model_means.shape
        batch_idxes = np.arange(0, batch_size)

        model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        samples = ensemble_samples[model_idxes, batch_idxes]  
        model_means = ensemble_model_means[model_idxes, batch_idxes]
        model_stds = ensemble_model_stds[model_idxes, batch_idxes]
        rewards, next_obs = samples[:, :1], samples[:, 1:]

        if return_single:
            next_obs = next_obs[0] 
            rewards = rewards[0]

        return next_obs, rewards

    def rollout_H(self, obs, agent, H=10, deterministic=False):
        assert H>=1
        s_0 = obs.copy()
        s = s_0.copy()
        reward_rollout = []
        len_rollout = 0
        for ii in range(H):
            act = agent.select_action(s)
            if ii==0:
                a_0 = act.copy()
            next_s, rewards = self.step(s, act)
            reward_rollout.append(rewards)
            len_rollout+=1
            s = next_s.copy()
        s_H = next_s
        a_H = agent.select_action(s_H)
        return s_0, a_0, s_H, a_H, reward_rollout, len_rollout

    def rollout_H_tensor(self, obs, agent, H=10, deterministic=False):
        s_0 = obs.clone().detach()  
        s = s_0.clone()
        reward_rollout = []
        len_rollout = 0
        for ii in range(H):
            act,_,_ = agent.policy.sample(s)
            if ii==0:
                a_0 = act.clone()
            next_s, rewards = self.step_tensor(s, act)
            reward_rollout.append(rewards)
            len_rollout+=1
            s = next_s.clone()
        s_H = next_s
        a_H,_,_ = agent.select_action(s_H)
        return s_0, a_0, s_H, a_H, reward_rollout, len_rollout

    def update(self, env_pool, batch_size, weight_grad_loss, near_n):
        state, action, reward, next_state, mask, done = env_pool.sample(len(env_pool))
        delta_state = next_state - state
        inputs = np.concatenate((state, action), axis=-1)
        labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)

        state_regular, action_regular, reward_regular, next_state_regular, _, _ = env_pool.sample_near(min(len(env_pool),10000)) 

        self.model.train(inputs, labels, state_regular, action_regular, next_state_regular, reward_regular, weight_grad_loss=weight_grad_loss, batch_size=256, holdout_ratio=0.2, near_n=near_n)

def get_batch_jacobian(net, x, noutputs): # x: b, in dim, noutpouts: out dim
    x = x.unsqueeze(1) # b, 1 ,in_dim
    n = x.size()[0]
    x = x.repeat(1, noutputs, 1) # b, out_dim, in_dim
    x.requires_grad_(True)
    y = net(x)
    input_val = torch.eye(noutputs).reshape(1,noutputs, noutputs).repeat(n, 1, 1)
    re = torch.autograd.grad(y,x,input_val, create_graph=True)[0]

    return re