import gym 
import numpy as np
env = gym.make("MountainCar-v0")
from OpenGL.GL import *

learning_rate = 0.1
discount = 0.95
episodes = 20000

show_every = 2000

# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.n) 

# print(len(env.observation_space.high))
# exit()



#bining discrete the size states 
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)

discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

epsilon = 0.5
start_epsilon_decaying = 1
end_epsilon_decaying = episodes // 2
eplison_decay_value = epsilon/(end_epsilon_decaying - start_epsilon_decaying) 
# print(discrete_os_win_size) 

# build Q table 

q_table = np.random.uniform(low= -2, high= 0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))

# print(q_table.shape)
# print (q_table)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/ discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


# print(discrete_state) # position and velocity 
# print(q_table[discrete_state])  # q value for all the three action 
# print(np.argmax(q_table[discrete_state]))  # best action to perform 




# print(DISCRETE_OS_SIZE)


for episode in range(episodes):
    if episode %show_every == 0:
        print (episode)
        render = True
    else:
        render = False    

    
    discrete_state = get_discrete_state(env.reset())
    done= False
    while not done:
        if np.random.random()> epsilon:
            action = np.argmax(q_table[discrete_state]) 
        else:
            action =np.random.randint(0,env.action_space.n)    
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        #   print(reward,new_state)
        if render:
          env.render()
        if not done:
            max_future_q  = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1- learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action, )] = 0 


        discrete_state = new_discrete_state  
    if end_epsilon_decaying >= episode >= start_epsilon_decaying:
        epsilon -= eplison_decay_value       
env.close() 
#print(reward)

