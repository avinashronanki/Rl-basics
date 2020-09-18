import gym 
import numpy as np
env = gym.make("MountainCar-v0")
env.reset()

# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.n) 

# print(len(env.observation_space.high))

#bining discrete the size states 
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)

discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
print(discrete_os_win_size) 


# build Q table 

q_table = np.random.uniform(low= -2, high= 0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))

print(q_table.shape)
print (q_table)


# print(DISCRETE_OS_SIZE)

done= False
while not done:
  action = 2 #push the car right 
  new_state, reward, done, _ = env.step(action)
#   print(reward,new_state)
#   env.render()
env.close() 
#print(reward)





