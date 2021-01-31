import numpy as np 
import sys
import numpy as np
import random 
import pdb
import time
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

iteration_flag = False

def evaluation(env, Q_table, idx, q_iterations, epsilon, step_current_bound = 80, num_itr = 5):

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
        
            action_current,_,reduction_factor = get_action_current_egreedy(Q_table[state_current], epsilon, idx, q_iterations)
            r, state_current_n, done = env.step(state_current,action_current)
            state_current = state_current_n
            reward_current += r
            step_current +=1
        total_reward_current += reward_current
        total_step_current += step_current
        itr += 1
        
    # returns (1) the average number of steps in episode until goal is reached and average total reward per episode     
    return total_step_current/float(num_itr), total_reward_current/float(num_itr)

import numpy as np
import random 
import pdb
import time

action_dic ={ 0:"UP", 1:"DOWN", 2:"LEFT", 3:"RIGHT"} # ACTMAP gives our definition of "clockwise slipping". 
# That is, we meant to go 0:UP, but instead we go 3:Right. Etc.
ACTMAP = {0:3, 1:2, 2:0, 3:1}
color2num = dict(  gray=30, red=31, green=32, yellow=33, blue=34, magenta=35, cyan=36, white=37, crimson=38)
aa = 0
class Maze():
    # state ID : 0, ..., 111
    # action ID : 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT
    obstacles = [(0,1),(0,3),(2,0),(2,4),(3,2),(3,4)]
    def __init__(self):
        self.episodic = True
        self.stochastic = True
        self.snum = 112
        self.anum = 4
        self.slip = 0.1
        self.dim = (4,5)
        self.start_pos = (0,0)
        self.goal_pos = (0,4)
        self.goal = (96,104)
        self.gamma = 0.9  
        self.map = np.asarray(["SWFWG","OOOOO","WOOOW","FOWFW"])
        self.img_map = np.ones(self.dim)
        for x in Maze.obstacles:
            self.img_map[x[0]][x[1]] = 0
        self.idx2cell = {0: (0, 0), 1: (1, 0), 2: (3, 0), 3: (1, 1), 4: (2, 1), 5: (3, 1),
            6: (0, 2), 7: (1, 2), 8: (2, 2), 9: (1, 3), 10: (2, 3), 11: (3, 3), 12: (0, 4), 13: (1, 4)}
        self.cell2idx = {(1, 2): 7, (0, 0): 0, (3, 3): 11, (3, 0): 2, (3, 1): 5, (2, 1): 4, 
            (0, 2): 6, (1, 3): 9, (2, 3): 10, (1, 4): 13, (2, 2): 8, (0, 4): 12, (1, 0): 1, (1, 1): 3}
    
    def step(self,state,action,default_slip=True,custom_slip =0):
        

        k = np.random.rand()
        
        if k < (self.slip * default_slip + custom_slip * (not default_slip)):
            a = ACTMAP[action]
            # TEST print("SLIP")
        else:
            a = action
            # TEST print("NO SLIP")

        # TEST print("SLIP THRESHOLD:",  (self.slip * default_slip + custom_slip * (not default_slip)))
        
        cell = self.idx2cell[int(state/8)] 
        if a == 0:
            c_next = cell[1]
            r_next = max(0,cell[0]-1)
        elif a ==1:
            c_next = cell[1]
            r_next = min(self.dim[0]-1,cell[0]+1)
        elif a == 2:
            c_next = max(0,cell[1]-1)
            r_next = cell[0]
        elif a == 3:
            c_next = min(self.dim[1]-1,cell[1]+1)
            r_next = cell[0]
        else:
            print (action, a) 
            raise ValueError

        if (r_next == self.goal_pos[0]) and (c_next == self.goal_pos[1]): # Reach the exit row and column
            v_flag = self.num2flag(state%8) # the flag tuple is just the remainder of the modulo (remainder) between state and 8
            
            # if we got to the end, the reward, return the number of flags captured, the final state and a "True" boolean variable
            return float(sum(v_flag)), 8*self.cell2idx[(r_next,c_next)] + state%8, True 

        else:
            # if we're going towards an obstacle...
            if (r_next,c_next) in Maze.obstacles: # obstacle tuple list
                #... we get 0.0 reward, stay in the same 'state', and finished = 'False' (we're not at goal)
                return 0.0, state, False
            else: # Flag locations
                
                # get the number of flags that are being held based on state
                v_flag = self.num2flag(state%8)

                # if we are going to any flag state, update v_flag tuple
                if (r_next,c_next) == (0,2):
                    v_flag[0] = 1
                elif (r_next,c_next)==(3,0):
                    v_flag[1] = 1
                elif (r_next,c_next) == (3,3):
                    v_flag[2] = 1
                
                # return 0.0 reward (we're not at goal), the next state, and finished = 'False' (we're not at goal)
                return 0.0, 8*self.cell2idx[(r_next,c_next)] + self.flag2num(v_flag), False

    def num2flag(self,n):
        # n is a positive integer
        # Each element of the below tuple correspond to a status of each flag. 0 for not collected, 1 for collected. 
        flaglist = [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]
        return list(flaglist[n])

    # It's hard to see this just by looking at this function, but it turns out that
    # this function is the inverse of the num2flag function, which is an easy function to understand
    def flag2num(self,v):
        # v: list
        if sum(v) < 2:
            return np.inner(v,[1,2,3])
        else:
            return np.inner(v,[1,2,3])+1

    def reset(self):
        # Return the initial state
        return 0

    def policy_evaluation(self,pi_0,v_init = True,value_array=0):

        # A list from which the possible actions are indexed: "UP","DOWN","LEFT","RIGHT"
        action_list = [0,1,2,3]

        # a list holding 112 observations with the following:
        ''' for 100% chance of no slip, 0.0% chance of slip  
            if I go: up, down, left or right
            the (reward, next state, done) are'''
        # every time the policy_evaluation function gets called, this list NEEDS TO BE INITIALIZED
        # this is because we are working with a new policy pi every time the first layer of this recursion gets called
        expectation_list = []

        # if this is the first layer of the recursion, set all cell values to zero
        if v_init:
            v_0 = np.zeros(self.snum)

        # if this is NOT the first layer of recursion, set all cell values to the value_array input
        else:
#             print("RECURSION!")
            v_0 = value_array

        # list containing the transition probabilities into the next state based on slip probability 
        emission_coef_list = [1-self.slip,self.slip]
        # TEST: print(emission_coef_list,"emission_coef_list")
        
        # creates indexer for obtaining immediate reward + future rewards (from 0 to 111) 
        state_indexer = np.arange(self.snum)
        # TEST: print(state_indexer,"state_indexer")
        
        # compute immediate and future reward based on no slip
        for state_given in state_indexer: 
            
            if state_given not in range(96,104):
                reward_list_ns = [self.step(state_given, action_t, False,0) for action_t in action_list]
                reward_list_s  = [self.step(state_given, action_t, False,1) for action_t in action_list]

            else:
                reward_list_ns = [(0,state_given,True) for action_t in action_list]
                reward_list_s  = [(0,state_given,True) for action_t in action_list]    
            
            expectation_list.append([reward_list_ns,reward_list_s])
            
            
        expectation_input_array = np.asarray(expectation_list)

        reward_array_ns     = np.asarray(expectation_input_array[:,0,:,0])
        next_state_array_ns = np.asarray(expectation_input_array[:,0,:,1]).astype(int)

        expected_rewards_ns = reward_array_ns + self.gamma * v_0[next_state_array_ns]
        
        expected_rewards_ns = emission_coef_list[0] * expected_rewards_ns

        reward_array_s     = np.asarray(expectation_input_array[:,1,:,0])
        next_state_array_s = np.asarray(expectation_input_array[:,1,:,1]).astype(int)

        expected_rewards_s  = reward_array_s + self.gamma * v_0[next_state_array_s]
        expected_rewards_s  = emission_coef_list[1] * expected_rewards_s
        
        expectation_array = expected_rewards_ns + expected_rewards_s
        
        action_value_pairs = np.copy(expectation_array)  
        
        expectation_array = np.multiply(pi_0,expectation_array)    
        expectation_array = np.sum(expectation_array,axis=1)
        
        if np.sqrt(np.sum(np.square(expectation_array - v_0))/(self.snum*self.anum)) >0.00000001:
            expectation_array,action_value_pairs = self.policy_evaluation(pi_0,False,expectation_array)
            
        return expectation_array,action_value_pairs

    def policy_improvement(self,action_value_pairs,idx):

        pi_improved = np.zeros((self.snum,self.anum))
        
        q_star      = np.copy(action_value_pairs)
        q_star_max  = np.amax(q_star,axis = 1)

        pi_indexer = np.argmax(action_value_pairs,axis=1)

        for idx,state_given in enumerate(pi_improved):
            state_given[pi_indexer[idx]] = 1
            q_star[idx][q_star[idx] < q_star_max[idx]] = 0
        

        return pi_improved,q_star

    def policy_iteration(self):
        
        iterations = 5
        
        pi = 1/self.anum * np.random.rand(self.snum,self.anum)
   
        for idx in range(iterations):
            action_value_pairs = self.policy_evaluation(pi)[1]
            pi_old = pi
            pi,q_star   = self.policy_improvement(action_value_pairs,idx)
        
        np.save('action_value_pairs.npy', action_value_pairs)       
        return pi,action_value_pairs

    def q_learning(self,loss_arange_par,q_iterations_par,adaptive_lr = 0,step_penalty = 0,epsilon_module = 0,epsilon_reduction_factor=0):
        
        print_review = False                # print comments for testing
        q_matrices_learning_rates = []     # hold the q matrixes values for different learning rates
        RMSE_learning_rates = []                     # hold the root mean square error for different learning rates
        evaluation_learning_rates = []        # captures the output of the evaluation funcition over learning rates
        
        for lr in loss_arange:                         # for the differente training rates tested  
            print("Learning Rate:", lr)                # print the learning rate
            start = time.time()                        # start counting time
            q_matrix_iterations = []                   # captures q matrix for each iteration
            RMSE_iterations = []                       # captures RMSE for each iteration
            evaluation_interations = []                # captures the output of the evaluation funcition each 50 steps

            q_matrix  = np.zeros([self.snum,self.anum])  # initialize state values
            state_t = 0                                  # initialize initial state
            state_next = 0
            done = False
            action_t = np.random.randint(4)
            idx = 0
            steps = 5000
            
            while idx < steps:
                                
                epsilon = (1-idx/steps*epsilon_reduction_factor)
                action_t,_,_ = get_action_current_egreedy(q_matrix[state_next], epsilon, idx, q_iterations)
                
                reward_next, state_next, done = self.step(state_t, action_t)
                state_next = int(state_next)
                
                q_max = np.amax(q_matrix[state_next])    
                q_matrix[state_t,action_t] += lr*(1-idx/q_iterations*adaptive_lr) * (reward_next + self.gamma * q_max - q_matrix[state_t,action_t])        
                                
                
                if done:
                    if idx % 50 == 0:
                        evaluation_interations.append(evaluation(self,q_matrix,idx, q_iterations,epsilon_module*(1-idx/q_iterations)))        
                        RMSE_iterations.append(np.sqrt(np.sum(np.square(q_matrix-action_value_pairs))/(self.snum*self.anum))) 
                    state_t = 0
                    idx +=1
                    if idx % 500 == 0:
                        print(idx,"Steps out of",steps,"completed.")
                else:
                    state_t = int(state_next)
                
                q_matrix_iterations.append(q_matrix)
                   
                
            q_matrices_learning_rates.append(q_matrix_iterations)
            RMSE_learning_rates.append(RMSE_iterations)
            evaluation_learning_rates.append( evaluation_interations)                
        
        
        return q_matrices_learning_rates,RMSE_learning_rates,evaluation_learning_rates,self.slip

### Simple Testing code added by Travers
if __name__ == '__main__':
    maze = Maze()
    # This test shows that maze.flag2num is indeed the inverse of maze.num2flag
    for i in range(8):
        np.testing.assert_almost_equal(i, maze.flag2num(maze.num2flag(i)))

### Example main method to allow interaction with the environment
if __name__ == '__main__':
    maze = Maze()

    state = maze.reset()

def get_action_current_egreedy(q_values_for_state,epsilon,idx,q_iterations):
    
    if np.random.rand() < epsilon:
        action = np.random.choice(4)
    else:
        action = np.argmax(q_values_for_state)
    return action, 1,0

if True or not iteration_flag:
    env = Maze() 
    initial_state = env.reset() 
    pi, action_value_pairs = env.policy_iteration()
    iteration_flag = True
    
print(action_value_pairs)


eval_steps, eval_reward = [], [] 
learning = True 
env = Maze() 
initial_state = env.reset()

start = .5
loss_arange = np.arange(start,2*start+.01,.1)

a_lr = 1
epsilon_factor_evaluation = 0.00
epsilon_reduction_factor_exploration = 0
q_iterations = 5000
step_penalty = 0
for _ in range(1):
    
    q_matrix_interations,RMSE_list,evaluation_learning_rates_list,slip =                    env.q_learning(loss_arange,q_iterations,a_lr,step_penalty,epsilon_factor_evaluation,epsilon_reduction_factor_exploration)

    fig1 = plt.figure(figsize=(15,15))
    indexer= np.arange(0,q_iterations,50)

    for i,j in enumerate(evaluation_learning_rates_list):
        
        plt.plot(indexer,np.asarray(j)[:,1], alpha = (i+1)/len(evaluation_learning_rates_list))
        plt.scatter(indexer,np.asarray(j)[:,1],label=loss_arange[i], alpha = (i+1)/len(evaluation_learning_rates_list))

        plt.legend(title="Slip = "+str(slip)+"\nStep Penalty = "+str(step_penalty)+"\nAdaptive Learning Rate = "+str(a_lr)+"\nEpsilon Factor (Exploration) = "+str(epsilon_reduction_factor_exploration)+"\nEpsilon Factor (Evaluation) = "+str(epsilon_factor_evaluation)+"\n         Learning Rates:")

    plt.title("Average Reward Versus Iteration", size = 30)
    plt.xlabel("Iteration", size =20)
    plt.ylabel("Average Reward", size =20)
    plt.ylim(0,3)
    plt.savefig("q_learning_plots/evaluation_plots"+str(time.time())+".png",dpi=150)
    plt.show()


list_states = np.arange(112)

for i in range(len(q_matrix_interations)):
    for j,k in enumerate(q_matrix_interations[i][-1]):
        if j in list_states:
            print(j)
            print(action_value_pairs[j])
            print(k,"\n")


fig1 = plt.figure(figsize=(15,15))

for idx,i in enumerate(RMSE_list):
    plt.plot(indexer,i,label=loss_arange[idx])
    plt.legend(title="Slip = "+str(slip)+"\nStep Penalty = "+str(step_penalty)+"\nAdaptive Learning Rate = "+str(a_lr)+"\nEpsilon Factor (Exploration) = "+str(epsilon_reduction_factor_exploration)+"\nEpsilon Factor (Evaluation) = "+str(epsilon_factor_evaluation)+"\n         Learning Rates:")
plt.title("RMSE Loss Versus Iteration", size = 30)
plt.xlabel("Iteration", size =20)
plt.ylabel("RMSE Loss", size =20)
plt.savefig("q_learning_plots/learning_plots"+str(time.time())+".png",dpi=150)
plt.show()