
# coding: utf-8

# In[1]:


import gym
import time
import numpy as np
import matplotlib.pyplot as plt 
import imageio
import os


# In[2]:


car = gym.make('MountainCar-v0')   # make mountain car
car._max_episode_steps = 1000
car.reset()


# In[3]:


directory = 'mountain_car_figs/'

if not os.path.exists(directory):
    os.makedirs(directory)


# In[4]:


def get_action_current_egreedy(q_values_for_state,epsilon):
    
    if np.random.rand() < epsilon:
        action = np.random.choice(4)
    else:
        action = np.argmax(q_values_for_state)
    return action, 1,0


# In[5]:


def evaluation(env, Q_table, state_t_discrete, epsilon, scaling_array, step_current_bound = 500, num_itr = 20):
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
            
            state_discrete = discretize_state(state_current,scaling_array)
            action_current,_,reduction_factor = get_action_current_egreedy(Q_table[state_discrete[0],state_discrete[1]], epsilon)
            state_current_n, r, done, _ = env.step(action_current)
            state_current = state_current_n
            reward_current += r
            step_current +=1
        total_reward_current += reward_current
        total_step_current += step_current
        itr += 1
        
    # returns (1) the average number of steps in episode until goal is reached and average total reward per episode     
    return total_step_current/float(num_itr), total_reward_current/float(num_itr)


# In[6]:


def discretize_state(state,scaling_array):
    state_t_discrete = (state - car.observation_space.low)*scaling_array
    state_t_discrete = np.around(state_t_discrete, decimals = 0).astype(int)
    return state_t_discrete


# In[7]:


def q_learning(car, lr, gamma, epsilon, epsilon_floor, episode_cnt,print_count,time_cluster):
    
    # get parameter range
    parameter_range = car.observation_space.high - car.observation_space.low
    
    # scaling array for bins
    scaling_array = np.array([12, 80]) 
#     scaling_array = np.array([10, 50]) 

    # create state bin count
    state_cnt = scaling_array * parameter_range
    state_cnt = 1 + np.around(state_cnt, decimals = 0).astype(int)
    
    # q_matrix initialization
    q_matrix = - 0.0001 * (np.random.rand(state_cnt[0], state_cnt[1], car.action_space.n) - 0.5)
    
    # create lists to store output
    reward_list = []
    rewards_average_list = []
    evaluation_iterations =[]
    
    # compute reduction factor
    epsilon_reduction_factor = (epsilon - epsilon_floor)/episode_cnt

    # q learning algorithm
    for episode in range(episode_cnt):
        
        # init constants
        state_t = car.reset()
        episode_reward,reward_next = 0,0
        done = False
        render_list = []
        # initalize current discrete state
        state_t_discrete = discretize_state(state_t,scaling_array)
        
        # counts the number of steps in a given direction (since action was last changed)
        counter_dir = 0
        
        while not done:   
            
            # render the last n episodes, save to list
            if episode >= (episode_cnt - print_count):
                render_list.append(car.render(mode='rgb_array'))
                
            # choose action at time t based on e-greedy algorithm
            if np.random.random() < 1 - epsilon:
                action_t = np.argmax(q_matrix[state_t_discrete[0], state_t_discrete[1]]) 
            else:
                action_t = np.random.randint(0, car.action_space.n)
                
            # get next state, next reward, done
            state_next, reward_next, done, _ = car.step(action_t) 

            # discretize state_next
            state_next_discrete = discretize_state(state_next,scaling_array)
            
            # update q matrix for final state
            if done and state_next[0] >= 0.50:
#                 print("Reward After Done:", reward_next)
                q_matrix[state_t_discrete[0], state_t_discrete[1], action_t] = reward_next + 100
                
            # if not done, update q matrix for current state
            else:
                temporal_difference = lr*(1-episode/episode_cnt)*(reward_next + 
                                 gamma*np.amax(q_matrix[state_next_discrete[0], 
                                                   state_next_discrete[1]]) - 
                                 q_matrix[state_t_discrete[0], state_t_discrete[1],action_t])
                
                q_matrix[state_t_discrete[0], state_t_discrete[1],action_t] += temporal_difference
                                     
            # update state and reward
            state_t_discrete = state_next_discrete
            episode_reward +=reward_next
            
            counter_dir += 1
#             print("STATE", state_t)
            
            
        # apply epsilon reduction
        if epsilon > epsilon_floor:
            epsilon -= 2*epsilon_reduction_factor
        
        # add reward to eposide list
        reward_list.append(episode_reward)
        
        if (episode+1) % 50 == 0:
            epsilon_module = 0.0
            evaluation_iterations.append(evaluation(car, q_matrix, state_t_discrete, epsilon_module*(1-episode/episode_cnt),scaling_array)) 
        
        if (episode+1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            rewards_average_list.append(ave_reward)
            reward_list = []
            
        if (episode+1) % 1000 == 0:    
            print('Average Reward for Episode {} : {}'.format(episode+1, ave_reward))
     
    car.close()    
    
    
    
    if len(render_list) >0:
        
        directory = 'mc_gifs/'

        if not os.path.exists(directory):
            os.makedirs(directory)
        
        imageio.mimsave(directory+str(lr)+str(time.time())+'.gif', render_list, fps=30)
    return rewards_average_list,render_list,evaluation_iterations


# In[8]:


lr_list = np.arange(.2,.51,.1)
epsilon_floors = np.arange(0,0.01,.1)

gamma = 0.9
epsilon = 1
render_cnt = 0

episode_cnt = 20000
time_cluster = 1

for min_epsilon in epsilon_floors:
    
    rewards_lr = []
    renders_lr = []
    evaluation_lr = []
    
    for lr in lr_list:
        rewards,render_list,evaluation_interations = q_learning(car, lr, gamma,epsilon, min_epsilon, episode_cnt,render_cnt,time_cluster)
        rewards_lr.append(rewards)
        renders_lr.append(render_list)
        evaluation_lr.append(evaluation_interations)
        
    directory = 'mountain_car_exploration_rewards/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    # generate plot 
    fig1 = plt.figure(figsize=(15,15))
    for i,j in enumerate(lr_list):
        plt.plot(100*(np.arange(len(rewards_lr[i])) + 1), rewards_lr[i],label=str(j),alpha=(i+1)/len(lr_list))
        plt.legend(title="Time Cluster Size = "+str(time_cluster)+"\nGamma = "+str(gamma)+"\nEpsilon Initial (Exploration) = "+str(epsilon)+"\nEpsilon Minimum = "+str(min_epsilon)+"\n         Learning Rates:")
    plt.title("Average Reward vs Episode (Exploration Mode)", size = 30)
    plt.xlabel("Episode Number",size =30)
    plt.ylabel("Mean Reward",size =30)
    plt.savefig(directory+"rewards"+str(time.time())+".jpg")
    plt.show()
    plt.close()

    directory = 'mountain_car_evaluation_rewards/'

    if not os.path.exists(directory):
        os.makedirs(directory)


    # generate plot 
    fig1 = plt.figure(figsize=(15,15))
    for i,j in enumerate(lr_list):
        plt.plot(50*(np.arange(len(evaluation_lr[i])) + 1), np.asarray(evaluation_lr[i])[:,1],label=str(j),alpha=(i+1)/len(lr_list))

    plt.legend(title="Time Cluster Size = "+str(time_cluster)+"\nGamma = "+str(gamma)+"\nEpsilon Initial (Exploration) = "+str(epsilon)+"\nEpsilon Minimum = "+str(min_epsilon)+"\n         Learning Rates:")
    plt.title("Average Reward vs Episode (Evaluation Mode)", size = 30)
    plt.xlabel("Episode Number",size =30)
    plt.ylabel("Mean Reward",size =30)

    plt.savefig(directory+"rewards"+str(time.time())+".jpg")
    plt.show()
    plt.close()    

