import argparse
import numpy as np
import matplotlib.pyplot as plt
import itertools
import copy
import os
import time
import torch
import random
import math
import torch.nn as nn
import gym
from torch.utils.tensorboard import SummaryWriter
import datetime
from nets import Simple_model
from buffer import ReplayMemory        
from utility import xu2t
from agent_double_model import Agent
import env
parser = argparse.ArgumentParser(description='PyTorch agent')
#### sac
parser.add_argument('--env-name', default="MyPendulum-v1",
                    help='Gym environment (default: MyPendulum-v1)')
parser.add_argument('--seed', type=int, default=5, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                    help='model updates per simulator step (default: 5)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')                    
#### buffer
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
#### rollout like mbpo
parser.add_argument('--model_retain_epochs', type=int, default=1, metavar='A',
                    help='retain epochs')
parser.add_argument('--rollout_batch_size', type=int, default=100000, metavar='A',
                    help='rollout number M')
parser.add_argument('--epoch_length', type=int, default=1000, metavar='A',
                    help='steps per epoch')
parser.add_argument('--rollout_min_epoch', type=int, default=25, metavar='A',
                    help='rollout min epoch')
parser.add_argument('--rollout_max_epoch', type=int, default=155, metavar='A',
                    help='rollout max epoch')
parser.add_argument('--rollout_min_length', type=int, default=1, metavar='A',
                    help='rollout min length')
parser.add_argument('--rollout_max_length', type=int, default=15, metavar='A',
                    help='rollout max length')
#### model
parser.add_argument('--num_networks', type=int, default=7, metavar='E',
                    help='ensemble size (default: 7)')
parser.add_argument('--num_elites', type=int, default=5, metavar='E',
                    help='elite size (default: 5)')
parser.add_argument('--pred_hidden_size', type=int, default=200, metavar='E',
                    help='hidden size for predictive model')
parser.add_argument('--reward_size', type=int, default=1, metavar='E',
                    help='environment reward size')
parser.add_argument('--model_train_freq', type=int, default=250, metavar='A',
                    help='frequency of training')
parser.add_argument('--model_train_batch_size', type=int, default=256, metavar='N',
                    help='model train batch size (default: 256)')

parser.add_argument('--use_decay', type=bool, default=True, metavar='G',
                    help='decay for weight for update of model (default: True)')
parser.add_argument('--weight_grad', type=float, default=10, metavar='G',
                    help='weight_grad (default: 10)')
parser.add_argument('--near_n', type=int, default=5, metavar='N',
                    help='model train tree choose near n (default: 5)')
#### update policy
parser.add_argument('--batch_size_policy', type=int, default=16, metavar='N',
                    help='batch size (default: 16)')
parser.add_argument('--update_policy_times', type=int, default=5, metavar='N',
                    help='update_policy_times (default: 5)')

#### train setting
parser.add_argument('--max_train_repeat_per_step', type=int, default=5, metavar='A',
                    help='max training times per step')
parser.add_argument('--exploration_init', action="store_true",
                    help='init exploration (default: False)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--start_steps', type=int, default=5000, metavar='N',
                    help='Steps sampling random actions (default: 5000)')
parser.add_argument('--min_pool_size', type=int, default=5000, metavar='A',
                    help='minimum pool size')  ## start update use model or fake data
parser.add_argument('--real_ratio', type=float, default=0.05, metavar='A',
                    help='ratio of env samples / model samples')
parser.add_argument('--H', type=int, default=4, metavar='N',  
                    help='number rollout (default: 4)')  # different of paper, 3 here means 2 in paper
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--see_freq', type=int, default=1000, metavar='N',
                    help='see_freq, type=int, default=1000')
parser.add_argument('--efficient', type=bool, default=False, metavar='G',
                    help='update q and policy like sac when collecting data using random policy (default: False)')

args = parser.parse_args()                   
if not os.path.exists("./results"):
    os.makedirs("./results")
if not os.path.exists("./models"):
    os.makedirs("./models")
#############################
if args.cuda:
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed
    torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)


##############################
# Environment
env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

env_e = gym.make(args.env_name)
env_e.seed(1234)
env_e.action_space.seed(1234)

# Agent
agent = Agent(env.observation_space.shape[0], env.action_space, args)

def resize_model_pool(args, rollout_length, model_pool):
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch

    sample_all = model_pool.return_all()
    new_model_pool = ReplayMemory(new_pool_size, args.seed)
    new_model_pool.push_batch(sample_all)

    return new_model_pool

def set_rollout_length(args, epoch_step):
    rollout_length = (min(max(args.rollout_min_length + (epoch_step - args.rollout_min_epoch)
                              / (args.rollout_max_epoch - args.rollout_min_epoch) * (args.rollout_max_length - args.rollout_min_length),
                              args.rollout_min_length), args.rollout_max_length))
    return int(rollout_length)

def rollout_for_update_q(agent, memory, memory_fake, rollout_length=1, rollout_batch_size=100000):
    state, action, reward, next_state, mask, done = memory.sample_all_batch(rollout_batch_size)
    for i in range(rollout_length):
        # TODO: Get a batch of actions
        action = agent.select_action(state)
        next_states, rewards = agent.model_ensemble_rollout.step(state, action)
        # TODO: Push a batch of samples
        terminals = agent.termination_fn.done(state, action, next_states)
        memory_fake.push_batch([(state[j], action[j], rewards[j][0], next_states[j], float( not terminals[j]), terminals[j]) for j in range(state.shape[0])])
        nonterm_mask = ~terminals
        if nonterm_mask.sum() == 0:
            break
        state = next_states[nonterm_mask]

# Memory
memory = ReplayMemory(args.replay_size, args.seed)
print(int(args.rollout_batch_size * args.epoch_length / args.model_train_freq)*1*args.model_retain_epochs)
memory_fake = ReplayMemory(int(args.rollout_batch_size * args.epoch_length / args.model_train_freq)*1*args.model_retain_epochs, args.seed)

# Training Loop
updates_q = 0
updates_q_like_sac = 0
total_numsteps = 0
updates = 0
num_episodes = 0

######### save   #######
reward_save = []
num_updates_pmp = 0

flag_model_trained = False  

epoch_step=-1
epoch_length=args.epoch_length
rollout_length = 1
for i_episode in itertools.count(1):

    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    
    while not done:
        if args.exploration_init and ( args.start_steps > total_numsteps):
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, evaluate=False)  

        if total_numsteps%epoch_length==0:
            epoch_step+=1
        
        next_state, reward, done, _ = env.step(action)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        memory.push(state, action, reward, next_state, mask, done)
        if done:   
            num_episodes +=1        
        state = next_state

        if len(memory) >= args.batch_size and total_numsteps<args.min_pool_size and args.efficient:
            for i in range(1):
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters_like_sac(memory, args.batch_size, updates_q_like_sac)
                updates_q_like_sac += 1            

        if total_numsteps % args.model_train_freq == 0 and len(memory) >= args.model_train_batch_size and args.real_ratio < 1.0 and total_numsteps>args.min_pool_size:   #### 2
            agent.update_parameters_ensemble_model(memory, args.model_train_batch_size, args.weight_grad, args.near_n)
            flag_model_trained =True

            new_rollout_length = set_rollout_length(args, epoch_step)
            if rollout_length != new_rollout_length:
                rollout_length = new_rollout_length
                memory_fake = resize_model_pool(args, rollout_length, memory_fake)

            rollout_for_update_q(agent, memory, memory_fake, rollout_length, args.rollout_batch_size)

        # ### train policy ###
        if len(memory) >= args.batch_size and flag_model_trained and len(memory) >= args.min_pool_size:
            for i in range(args.update_policy_times):
                loss_policy,dQds_norm = agent.update_parameters_policy(state, memory, memory_fake, args.H, args.batch_size_policy)
            num_updates_pmp+=args.update_policy_times

        if len(memory) >= args.batch_size and len(memory) > args.min_pool_size: 
            for i in range(args.updates_per_step):
                if updates_q > args.max_train_repeat_per_step*(total_numsteps - args.min_pool_size): 
                    break
                # Update q
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha,dd,ee,ff,gg = agent.update_parameters_q(memory, memory_fake, args.batch_size, updates_q, real_ratio=args.real_ratio)
                updates_q += 1
        
        if total_numsteps % 10000 ==0:
            torch.cuda.empty_cache()

        # evaluate
        if total_numsteps % 1000 ==0 or total_numsteps==1:
            avg_reward = 0.
            avg_steps = 0.
            episodes = 10
            for _ in range(episodes): 
                episode_reward_e = 0
                episode_steps_e = 0
                done_e = False
                state_e = env_e.reset()
                while not done_e:
                    action_e = agent.select_action(state_e, evaluate=True)
                    episode_steps_e += 1
                    next_state_e, reward_e, done_e, _ = env_e.step(action_e)   #fix bug
                    episode_reward_e += reward_e
                    state_e = next_state_e
                avg_reward += episode_reward_e
                avg_steps += episode_steps_e
            avg_reward /= episodes
            avg_steps /= episodes
            reward_save.append([total_numsteps,avg_reward])
            print(total_numsteps,avg_reward)                          

    if total_numsteps > args.num_steps:
        break
env.close()
