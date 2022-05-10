import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from nets import DeterministicPolicy
from nets import GaussianPolicy
from nets import QNetwork
from utility import soft_update, hard_update
import copy
from model import EnsembleEnv as EnsembleEnv
from model_regular_on_jacobian import EnsembleEnv as EnsembleEnvRegular

class Termination_Fn(object):
    def __init__(self, env_name):
        self.env_name = env_name
        print(self.env_name)
        
    def done(self, obs, act, next_obs):
        if self.env_name=='HalfCheetah-v2' or self.env_name=='Reacher-v2':
            done = np.array([False]).repeat(len(obs))
            # done = done[:,None]
            return done
        elif self.env_name=='Hopper-v2':
            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done =  np.isfinite(next_obs).all(axis=-1) \
                        * np.abs(next_obs[:,1:] < 100).all(axis=-1) \
                        * (height > .7) \
                        * (np.abs(angle) < .2)
            done = ~not_done
            # done = done[:,None]
            return done
        elif self.env_name=='Walker2d-v2':
            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done =  (height > 0.8) \
                        * (height < 2.0) \
                        * (angle > -1.0) \
                        * (angle < 1.0)
            done = ~not_done
            # done = done[:,None]
            return done
        elif self.env_name=='AntTruncatedObs-v2':
            x = next_obs[:, 0]
            not_done = 	np.isfinite(next_obs).all(axis=-1) \
                        * (x >= 0.2) \
                        * (x <= 1.0)

            done = ~not_done
            # done = done[:,None]
            return done
        elif self.env_name=='InvertedPendulum-v2':
            notdone = np.isfinite(next_obs).all(axis=-1) \
                    * (np.abs(next_obs[:,1]) <= .2)
            done = ~notdone

            # done = done[:,None]

            return done
        elif self.env_name=='HumanoidTruncatedObs-v2':
            z = next_obs[:,0]
            done = (z < 1.0) + (z > 2.0)

            # done = done[:,None]
            return done
        else:
            done = np.array([False]).repeat(len(obs))
            # done = done[:,None]
            return done


class Agent(object): 
    def __init__(self, num_inputs, action_space, args):
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.model_ensemble_rollout = EnsembleEnv(args.num_networks, args.num_elites, state_size=num_inputs, action_size=action_space.shape[0], use_decay=args.use_decay)
        self.model_ensemble_grad = EnsembleEnvRegular(args.num_networks, args.num_elites, state_size=num_inputs, action_size=action_space.shape[0], use_decay=args.use_decay)
 
        self.gamma = args.gamma
        self.gamma_tensor = torch.FloatTensor([args.gamma]).to(self.device)
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        self.action_space = action_space
        self.DIM_X = num_inputs
        self.DIM_U = action_space.shape[0]
        self.state_sequence = []
        self.p = []
        self.action_sequence = []

        self.loss_function = nn.MSELoss()
        self.flag=0
        self.memory = None
        self.termination_fn = Termination_Fn(args.env_name)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters_like_sac(self, memory, batch_size, updates):
        # Sample a batch from memory   update Q network
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, _ = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() 
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) 


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()


    def update_parameters_q(self, memory, memory_fake, batch_size, updates, use_decay=False, weight_decay=0.1, real_ratio=0.05):
        batch_real = int(batch_size * real_ratio)  
        if batch_size-batch_real>0 and len(memory_fake)>0: 
            state_batch_real, action_batch_real, reward_batch_real, next_state_batch_real, mask_batch_real, _ = memory.sample(batch_size=batch_real)
            state_batch_fake, action_batch_fake, reward_batch_fake, next_state_batch_fake, mask_batch_fake, _ = memory_fake.sample(batch_size=batch_size-batch_real)
            state_batch = np.concatenate((state_batch_real, state_batch_fake), axis=0)
            action_batch = np.concatenate((action_batch_real, action_batch_fake), axis=0)
            reward_batch = np.concatenate((reward_batch_real, reward_batch_fake), axis=0)
            next_state_batch = np.concatenate((next_state_batch_real, next_state_batch_fake), axis=0)
            mask_batch = np.concatenate((mask_batch_real, mask_batch_fake), axis=0)


            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        else:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch, _ = memory.sample(batch_size=batch_size)

            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        if use_decay:
            decay_loss=0
            for name, value in self.critic.named_parameters():
                decay_loss += weight_decay*torch.sum(torch.square(value))/2.
            qf_loss+=decay_loss
        self.critic_optim.zero_grad()
        qf_loss.backward()
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), clip_value=100.)
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]


        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        
        dd=0
        for name, value in self.critic.named_parameters():
            dd += torch.norm(value).item()
        
        ee = torch.norm(next_q_value/batch_size).item()

        ff=0
        for name, value in self.policy.named_parameters():
            ff += torch.norm(value).item()
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(),dd,ee,ff,torch.sum(next_state_log_pi).item()/batch_size
  
    def update_parameters_policy(self,state, memory, memory_fake, H=10, batch_size=256): 
        state_batch, _, _, _, _, _ = memory.sample(batch_size)
        state_batch = torch.from_numpy(state_batch).float().to(self.device).reshape(batch_size,self.DIM_X).requires_grad_()
        state_seq = []
        action_seq = []
        log_pi_seq = []
        done_seq = []
        p_seq = []
        H_seq = []
        not_done_list = [True]*state_batch.size(0)

        for i in range(H):
            action_batch, log_pi_batch, _ = self.policy.sample(state_batch)
            next_state_batch, reward_batch = self.model_ensemble_rollout.step_tensor(state_batch, action_batch)
            next_state_batch = next_state_batch.detach()
            done_batch = self.termination_fn.done(state_batch.detach().to('cpu').numpy(), action_batch.detach().to('cpu').numpy(), next_state_batch.detach().to('cpu').numpy())
            state_seq.append(state_batch)
            action_seq.append(action_batch)
            log_pi_seq.append(log_pi_batch)
            done_seq.append(done_batch)
            not_done_list = [not_done_list[k] and (not done_batch[k]) for k in range(state_batch.size(0))]
            state_batch = next_state_batch.clone().requires_grad_()

        max_length_sequence = H
        mask_batch_np = np.zeros([batch_size, max_length_sequence,1])  #### 1 if not done
        for i in range(batch_size):
            for j in range(H):
                mask_batch_np[i, j, 0] = 1
                if done_seq[j][i]:                
                    break
        mask_batch = torch.from_numpy(mask_batch_np).to(self.device).reshape(batch_size,-1,1)

        state_input = state_seq[-1]
        action_input = action_seq[-1]
        log_pi = log_pi_seq[-1]

        qf1, qf2 = self.critic(state_input, action_input)
        min_qf = torch.min(qf1, qf2)
        if self.automatic_entropy_tuning:
            self.alpha = self.log_alpha.exp()
        L = (-min_qf + self.alpha * log_pi).view(batch_size,1) * ( self.gamma ** ( torch.sum(mask_batch,1)-1) ).view(batch_size,1) * mask_batch[:,-1,0].view(batch_size,1)
        H = L
        g = torch.autograd.grad(torch.sum(L), state_input, retain_graph=True)[0].detach().view(batch_size, self.DIM_X)

        cc = (torch.sum(g**2)/batch_size).item()

        p_seq.append(g.clone())
        H_seq.append(H)

        for j in range(max_length_sequence-2,-1,-1 ):
            p_tmp = g.clone()
            state_input = state_seq[j]
            action_input = action_seq[j]
            log_pi = log_pi_seq[j]
            next_state, L = self.model_ensemble_grad.step_tensor(state_input, action_input)
            H = (torch.sum(p_tmp * next_state,1).view(batch_size,1) - L + self.alpha * log_pi) * ( self.gamma ** ( torch.sum(mask_batch[:,0:j+1,],1)-1 ) ).view(batch_size,1) * mask_batch[:,j,0].view(batch_size,1)
            H_seq.append(H)
            g = torch.autograd.grad(torch.sum(H), state_input, retain_graph=True)[0].detach().view(batch_size, self.DIM_X)
            p_seq.append(g.clone())
        p_seq = p_seq.reverse()

        self.policy_optim.zero_grad()
        loss = torch.sum(torch.stack(H_seq,0))/batch_size
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), clip_value=5.)
        self.policy_optim.step()  

        return loss.item(), cc  

    def update_parameters_ensemble_model(self, memory, batch_size, weight_grad, near_n):
        self.model_ensemble_rollout.update(memory, batch_size,weight_grad, near_n)
        self.model_ensemble_grad.update(memory, batch_size,weight_grad, near_n)

