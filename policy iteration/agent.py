import numpy as np
from visualize_train import draw_value_image, draw_policy_image

# left, right, up, down
ACTIONS = [np.array([0, -1]),
           np.array([0, 1]),
           np.array([-1, 0]),
           np.array([1, 0])]


class AGENT:
    def __init__(self, env, is_upload=False):

        self.ACTIONS = ACTIONS
        self.env = env
        HEIGHT, WIDTH = env.size()
        self.state = [0,0]

        if is_upload:
            dp_results = np.load('./result/dp.npz')
            self.values = dp_results['V']
            self.policy = dp_results['PI']
        else:
            self.values = np.zeros((HEIGHT, WIDTH))
            self.policy = np.zeros((HEIGHT, WIDTH,len(self.ACTIONS)))+1./len(self.ACTIONS)




    def policy_evaluation(self, iter, env, policy, discount=1.0):
        HEIGHT, WIDTH = env.size()
        post_value_table = np.zeros((HEIGHT, WIDTH),dtype=float)
        new_state_values = np.zeros((HEIGHT, WIDTH),dtype=float)
        iteration = 0
        while (True):
            for i in range(HEIGHT):
                for j in range(WIDTH):
                    if i == j and (( i == 0) or ( i == HEIGHT-1)):
                        value_t=0
                    else :
                        for act_value in range(4):
                            [i_,j_],reward = env.interaction([i,j],ACTIONS[act_value])
                            value_t = policy[i,j,act_value]*(reward + discount*post_value_table[i_,j_])
                    new_state_values[i,j] = value_t
            delta = post_value_table - new_state_values
            iteration+=1
            
            post_value_table = new_state_values
            if np.all(delta<=0.01):
                break


        draw_value_image(iter, np.round(new_state_values, decimals=2), env=env)
        return new_state_values, iteration





    def policy_improvement(self, iter, env, state_values, old_policy, discount=1.0):
        HEIGHT, WIDTH = env.size()
        policy = old_policy.copy()
        policy_stable = True
    

        for i in range(HEIGHT):
            for j in range(WIDTH):
                action_value=[]
                for act in ACTIONS:
                    [i_,j_],reward = env.interaction([i,j],act)
                    action_value.append(reward+discount*state_values[i_,j_])
                max_action = [action_v for action_v, x in enumerate(action_value) if x == max(action_value)] 
                for k in range(4):
                    policy[i,j,k] = 0
                for k in max_action:
                    policy[i,j,k] =1/len(max_action) 
                for k in range(4):
                    delta = old_policy - policy
                    if np.all(delta=0):
                        policy_stable = False
        


        print('policy stable {}:'.format(policy_stable))
        draw_policy_image(iter, np.round(policy, decimals=2), env=env)
        return policy, policy_stable

    def policy_iteration(self):
        iter = 1
        while (True):
            self.values, iteration = self.policy_evaluation(iter, env=self.env, policy=self.policy)
            self.policy, policy_stable = self.policy_improvement(iter, env=self.env, state_values=self.values,
                                                       old_policy=self.policy, discount=1.0)
            iter += 1
            if policy_stable == True:
                break
        np.savez('./result/dp.npz', V=self.values, PI=self.policy)
        return self.values, self.policy



    def get_action(self, state):
        i,j = state
        return np.random.choice(len(ACTIONS), 1, p=self.policy[i,j,:].tolist()).item()


    def get_state(self):
        return self.state

