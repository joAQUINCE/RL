
# coding: utf-8

# In[1]:


import gym
import time
import numpy as np
import matplotlib.pyplot as plt 
import imageio
import os


# In[2]:


acro = gym.make('Acrobot-v1')   # make acrobot
acro.reset() # reset state


# In[3]:


def get_action_current_egreedy(q_values_for_state,epsilon):
    
    if np.random.rand() < epsilon:
        action = np.random.choice(4)
    else:
        action = np.argmax(q_values_for_state)
    return action, 1,0


# In[4]:


def evaluation(env, Q_table, state_t_discrete, epsilon, scaling_array, step_current_bound = 300, num_itr = 5):
    total_step_current = 0
    total_reward_current = 0
    itr = 0
    while(itr<num_itr):
        step_current = 0
        np.random.seed()
        state_current = env.reset()
        reward_current = 0.0
        done = False

        while((not done) and (step_current < step_current_bound)):
            
            state_discrete = discretize_state(env,state_current,scaling_array)
            action_current,_,reduction_factor = get_action_current_egreedy(Q_table[ state_discrete[0],                                                                                    state_discrete[1],                                                                                    state_discrete[2],                                                                                    state_discrete[3]], epsilon)
            state_current_n, r, done, _ = env.step(action_current)
            state_current = state_current_n
            reward_current += r
            step_current +=1
        total_reward_current += reward_current
        total_step_current += step_current
        itr += 1
        
    # returns (1) the average number of steps in episode until goal is reached and average total reward per episode     
    return total_step_current/float(num_itr), total_reward_current/float(num_itr)


# In[5]:


def discretize_state(env,state,scaling_array):
    
    theta_1 = np.arctan(state[1]/state[0])
    theta_2 = np.arctan(state[3]/state[2])
    
    state_t_discrete = np.array([theta_1,theta_2,state[4],state[5]])
    low_limits = np.array([0,0,env.observation_space.low[4],env.observation_space.low[5]])
    
    state_t_discrete = (state_t_discrete - low_limits)*scaling_array
    state_t_discrete = np.around(state_t_discrete, decimals = 0).astype(int)
    
    
    return state_t_discrete


# In[6]:


def q_learning(acro_object, lr, gamma, epsilon, epsilon_floor, episode_cnt,print_count,time_cluster):
    
    # get parameter range
    parameter_range = acro_object.observation_space.high - acro_object.observation_space.low
    parameter_range = np.array([2*np.pi,2*np.pi,parameter_range[4],parameter_range[5]]) 
    # bin count/(one unit of magnitude) for sinusoid variables
    sinusoid_bin_per_radian = 3

    # bin count/(one unit of magnitude) for angular velocity variables
    omega_bin_per_unit = 1

    scaling_array = np.array([sinusoid_bin_per_radian,                              sinusoid_bin_per_radian,                              omega_bin_per_unit,                              omega_bin_per_unit]) 

    # create state bin count
    state_cnt = scaling_array * parameter_range
    state_cnt = 1 + np.around(state_cnt, decimals = 0).astype(int)
    print(state_cnt)
    # q_matrix initialization
    q_matrix = 0.0001 * (np.random.rand(state_cnt[0],                                        state_cnt[1],                                        state_cnt[2],                                        state_cnt[3], acro_object.action_space.n) - 0.5)
    
    # create lists to store output
    reward_list = []
    rewards_average_list = []
    evaluation_iterations =[]
    
    # compute reduction factor
    epsilon_reduction_factor = (epsilon - epsilon_floor)/episode_cnt

    # q learning algorithm
    for episode in range(episode_cnt):
        
        # init constants
        state_t = acro_object.reset()
        episode_reward,reward_next = 0,0
        done = False
        render_list = []
        # initalize current discrete state
        state_t_discrete = discretize_state(acro_object, state_t, scaling_array)
        
        # counts the number of steps in a given direction (since action was last changed)
        counter_dir = 0

        while not done:   
            
            # render the last n episodes, save to list
            if episode >= (episode_cnt - print_count):
                render_list.append(acro_object.render(mode='rgb_array'))
                
            # choose action at time t based on e-greedy algorithm

            if int(counter_dir) % time_cluster == 0:
                if np.random.random() < 1 - epsilon:
                    action_t = np.argmax(q_matrix[state_t_discrete[0],                                                  state_t_discrete[1],                                                  state_t_discrete[2],                                                  state_t_discrete[3]]) 
                else:
                    action_t = np.random.randint(0, acro_object.action_space.n)
                
            # get next state, next reward, done
            state_next, reward_next, done, _ = acro_object.step(action_t) 
            
#             theta_1 = np.arctan(state_next[1]/state_next[0])
#             theta_2 = np.arctan(state_next[3]/state_next[2])

#             state_next_radians = np.array([theta_1,theta_2,state_next[4],state_next[5]]) 
            
            # discretize state_next
            state_next_discrete = discretize_state(acro_object, state_next,scaling_array)
            
            # update q matrix for final state
            if done:
                q_matrix[   state_t_discrete[0],                            state_t_discrete[1],                            state_t_discrete[2],                            state_t_discrete[3], action_t] = reward_next
                
            # if not done, update q matrix for current state
            else:
                temporal_difference = lr*(1-(episode/episode_cnt)**2)*(reward_next + 
                                 gamma*np.amax(q_matrix[state_next_discrete[0],\
                                                        state_next_discrete[1],\
                                                        state_next_discrete[2],\
                                                        state_next_discrete[3]])\
                                                                  -\
                                             q_matrix[  state_t_discrete[0],\
                                                        state_t_discrete[1],\
                                                        state_t_discrete[2],\
                                                        state_t_discrete[3],action_t])
                
                q_matrix[   state_t_discrete[0],                            state_t_discrete[1],                            state_t_discrete[2],                            state_t_discrete[3],action_t] += temporal_difference
                                     
            # update state and reward
            state_t_discrete = state_next_discrete
            episode_reward +=reward_next
            counter_dir += 1
#             print("STATE", state_t)
            
            
        # apply epsilon reduction
        if epsilon > epsilon_floor:
            epsilon -= epsilon_reduction_factor
        
        # add reward to eposide list
        reward_list.append(episode_reward)
        
        if (episode+1) % 50 == 0:
            epsilon_module = 0.0
            evaluation_iterations.append(evaluation(acro_object, q_matrix, state_t_discrete, epsilon_module*(1-episode/episode_cnt),scaling_array)) 
        
        if (episode+1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            rewards_average_list.append(ave_reward)
            reward_list = []
            
        if (episode+1) % 1000 == 0:    
            print('Average Reward for Episode {} : {}'.format(episode+1, ave_reward))
     
    acro_object.close()  
    
    directory = 'acro_gifs/'

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if len(render_list) >0:
        imageio.mimsave(directory+str(lr)+str(time.time())+'.gif', render_list, fps=30)
    return rewards_average_list,render_list,evaluation_iterations


# In[7]:


# Run Q-learning algorithm

lr_list = np.arange(.1,.2,.02)
epsilon_floors = np.arange(0.0,.51,.125)


gamma = 0.9
epsilon = 1
render_cnt = 1

time_cluster = 1
episody_cnt = 10000

for min_epsilon in epsilon_floors:
    
    rewards_lr = []
    renders_lr = []
    evaluation_lr = []
    
    for lr in lr_list:
        rewards,render_list,evaluation_interations = q_learning(acro, lr, gamma,epsilon, min_epsilon, episody_cnt,render_cnt,time_cluster)
        rewards_lr.append(rewards)
        renders_lr.append(render_list)
        evaluation_lr.append(evaluation_interations)
        
    directory = 'acro_exploration_rewards/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    # generate plot 
    fig1 = plt.figure(figsize=(15,15))
    for i,j in enumerate(lr_list):
        plt.plot(100*(np.arange(len(rewards_lr[i])) + 1), rewards_lr[i],label=str(j),alpha=(i+1)/len(lr_list))
        plt.legend(title="Gamma = "+str(gamma)+"\nEpsilon Initial (Exploration) = "+str(epsilon)+"\nEpsilon Minimum = "+str(min_epsilon)+"\n         Learning Rates:")
    plt.title("Average Reward vs Episode (Exploration Mode)", size = 30)
    plt.xlabel("Episode Number",size =30)
    plt.ylabel("Mean Reward",size =30)
    plt.savefig(directory+"rewards"+str(time.time())+".jpg")
    plt.show()
    plt.close()

    directory = 'acro_evaluation_rewards/'

    if not os.path.exists(directory):
        os.makedirs(directory)


    # generate plot 
    fig1 = plt.figure(figsize=(15,15))
    for i,j in enumerate(lr_list):
        plt.plot(100*(np.arange(len(evaluation_lr[i])) + 1), np.asarray(evaluation_lr[i])[:,1],label=str(j),alpha=(i+1)/len(lr_list))

    plt.legend(title="Time Cluster Size"+str(time_cluster)+"\nGamma = "+str(gamma)+"\nEpsilon Initial (Exploration) = "+str(epsilon)+"\nEpsilon Minimum = "+str(min_epsilon)+"\n         Learning Rates:")
    plt.title("Average Reward vs Episode (Evaluation Mode)", size = 30)
    plt.xlabel("Episode Number",size =30)
    plt.ylabel("Mean Reward",size =30)

    plt.savefig(directory+"rewards"+str(time.time())+".jpg")
    plt.show()
    plt.close()    

