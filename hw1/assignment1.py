import numpy as np
import matplotlib.pyplot as plt
import time

class Bandit:
    def __init__(self, init = 0, drift = .0001):
        self.reward = init
        self.drift = drift
        self.move_mean_reward()

    def move_mean_reward(self):
        self.reward += np.random.normal(0, self.drift,1)[0]
        return

    def get_noisy_reward(self):
        return self.reward  + np.random.randn()

    def get_true_reward(self):
        return self.reward
    
    def reset_bandit(self):
        self.reward = 0

class ActionValueMethod:
    def __init__(self,bandits = [], step_rule = 0, 
                            step_size = 0.1, epsilon = 0.1):
        self.method   = ["sample", "constant"][step_rule]
        self.bandits  = bandits
        self.num_arms = len(self.bandits)
        self.alpha    = step_size
        self.epsilon  = epsilon
        self.isGreedy = True
        self.action   = 0
        self.visits   = [0]*self.num_arms
        self.Qvalues  = [0]*self.num_arms
        self.true_values  = [0]*self.num_arms
        self.step_reward   = 0
        self.optimal_step  = 0
        self.running_reward = 0
        self.running_opt_rate = 0

    def reset_arrays(self):
        self.visits   = [0]*self.num_arms
        self.Qvalues  = [0]*self.num_arms
        self.true_values = [0]*self.num_arms
        # also reset the bandits
        for b in self.bandits:
            b.reset_bandit()
            # print("\nCheck True: " , b.get_true_reward())
            # print("Check Noisy: " , b.get_noisy_reward(),"\n")

    def run_simulate(self, total_steps = 10000):
        
        step = 0 
        avg_reward = []
        optimal_rate = []

        while (step < total_steps):
            
            self.select_action() ## Epsilon-Greedy
            if self.method == "sample":
                self.step_reward = self.sample_average_update()
            elif self.method == "constant":
                self.step_reward = self.constant_update()
            step += 1
            self.optimal_step = (np.argmax(self.true_values) == self.action)
            self.running_reward = (self.step_reward + (step-1)*self.running_reward) / step
            self.running_opt_rate = (self.optimal_step + (step-1)*self.running_opt_rate) / step
            avg_reward.append(self.running_reward)
            optimal_rate.append(self.running_opt_rate)

            ## DEBUG : ensure updates occur properly ---------------------------
            # if (step-1 % 500 == 0):
            #     print("Action at step {} was: {} | Check true: {}".
            #                     format(go, self.action, np.argmax(self.true_values)))
            #     print("Action Was Greedy?: ", self.isGreedy)
            #     print("Alpha is updating? : ", self.alpha, 1/self.visits[self.action])
            #     print("Optimality Percentage  {}".format(np.mean(optimal)))
            #     print("\nCheck Q-Value Estimates : {}".format(self.Qvalues))
            #     print("\nCheck Visits : {} \n".format(self.visits))
            #     print("\nStep {} True Rewards : {}".format( step, np.round(self.true_values, 4)))
            
        
        return (avg_reward , optimal_rate)

    def select_action(self):
        ''' Implements epsilon-greedy for now. 
                Later, extend to include other strategies'''
        if np.random.rand() < self.epsilon:
            self.action = np.random.randint(self.num_arms)
            self.isGreedy = False
        else:
            self.action = np.argmax(self.Qvalues)
            self.isGreedy = True
        return

    def update_bandits(self):
        ''' Incremental equation. Bandits drift before next step'''
        R = self.bandits[self.action].get_noisy_reward()
        self.Qvalues[self.action] += self.alpha * \
                                    ( R - self.Qvalues[self.action] )    
        for n,bandito in enumerate(self.bandits):
            self.true_values[n] = bandito.get_true_reward()
            bandito.move_mean_reward()
        return R
    
    def sample_average_update(self):
        self.visits[self.action] = self.visits[self.action] + 1
        self.alpha = 1 / (self.visits[self.action])
        return self.update_bandits()
    
    def constant_update(self): 
        return self.update_bandits()
        

def plot_individual(data , title = "Default"):

    t = np.linspace(0,len(data),len(data))
    fig, ax = plt.subplots()
    plt.plot(t,data)
    ax.set_title(title)
    ax.set_ylabel("Avg Reward")
    ax.set_xlabel("Step")
    
    return fig,ax

def plot_overlay_avg_reward(data1 , data2 ):
    colors = ["tab:blue" , "orange"]
    labels = ["Sample Average Method" , "Constant Method"]
    fig, ax = plt.subplots()
    t = np.linspace(0,len(data1),len(data1))
    ax.plot(t, data1, color = colors[0])
    ax.plot(t, data2, color = colors[1])
    ax.set_ylabel("Avg Reward")
    ax.set_xlabel("Step")
    ax.set_title(r'Comparison, $\epsilon$ = 0.1')
    plt.legend(labels)

def plot_overlay_optimality_rate(data1 , data2 ):
    colors = ["tab:blue" , "orange"]
    labels = ["Sample Average Method" , "Constant Method"]
    fig, ax = plt.subplots()
    t = np.linspace(0,len(data1),len(data1))
    ax.plot(t, data1, color = colors[0])
    ax.plot(t, data2, color = colors[1])
    ax.set_ylabel("Optimality %")
    ax.set_xlabel("Step")
    ax.set_title(r'Comparison, $\epsilon$ = 0.1')
    plt.legend(labels)

def write_output(data_array, fname = "result.out"):
    # per the 4 line format required in PA#1
    # also feeds the visualize1.py file
    np.savetxt("result.out",data_array )
    return


if __name__ == "__main__":

    start_time = time.time()
    ## Set up test , num episodes, iterations per episode ------------------
    EPISODES = 300
    NUM_STEPS = 10000
    avg_data = np.zeros([EPISODES, NUM_STEPS])
    avg_optimality = np.zeros([EPISODES, NUM_STEPS])
    const_data = np.zeros([EPISODES, NUM_STEPS])
    const_optimality = np.zeros([EPISODES, NUM_STEPS])
    # Drift Factor ( test various size movements in random walk )
    sigma = 0.01
    # Exploratory factor epsilon
    EPSILON = 0.1
    results = np.zeros((4, NUM_STEPS), np.float32)  
    
    ## Method Options , Pass to ActionValue object -----------------------------
    #  0 : Sample Average Method 
    #  1 : Constant Stepsize Parameter
    ## others? ----------------------------------------------------------------
    sample  = ActionValueMethod(
                bandits = [Bandit(drift = sigma**2) for i in range(10)], 
                step_rule = 0 , step_size = 1 , epsilon = EPSILON 
            )
    constant = ActionValueMethod(
                bandits = [Bandit(drift = sigma**2) for i in range(10)], 
                step_rule = 1 , step_size = .01 , epsilon = EPSILON 
            )

    ## Run Sim once for samp_avg , then for const , then plot ---------------
    for i in range(EPISODES):
        rewards, optimals = sample.run_simulate(NUM_STEPS)
        avg_data[i,:] = rewards
        avg_optimality[i,:] = optimals
        sample.reset_arrays()
    # save 
    results[0,:] = np.mean(avg_data, axis=0)
    results[1,:] = np.mean(avg_optimality, axis=0)
    
    for i in range(EPISODES):
        rewards, optimals = constant.run_simulate(NUM_STEPS)
        const_data[i,:] = rewards
        const_optimality[i,:] = optimals
        constant.reset_arrays()
    # save 
    results[2,:] = np.mean(const_data, axis=0)
    results[3,:] = np.mean(const_optimality, axis=0)

    print(results[3,:])
    print(results[1,:])
  
    np.savetxt("result.out", results)
    print("\nTime it! Total: ", round(time.time() - start_time, 5), "\n")


    ### Extra Plotting Checks -================================================
    # fig1, ax1 = plot_individual(np.mean(avg_data, axis=0), 
    #                            title = "Sample Average Method")
    # fig2, ax2 = plot_individual(np.mean(const_data, axis=0), 
                                # title = "Constant Step Method")    
    #
    # plot_overlay_avg_reward(np.mean(avg_data, axis=0) , 
    #                                 np.mean(const_data, axis=0))
    # plot_overlay_optimality_rate(np.mean(avg_optimality,axis=0) , 
    #                                 np.mean(const_optimality,axis=0))
    #
    # plt.show()

