from DE import DE
import numpy as np
import ioh
import pickle
import gzip
import os
from ddqn import DDQNAgent

EPISODES = 10000
warmup = 5 # how often should each function have been visited before training starts,
           # i.e. how full should the memory be
batch_size = 512
tau = 10   # update target network every tau episodes


if __name__ == "__main__":
    # Initialise objects
    func_choice = [ioh.problem.BBOB.problems[func] for func in ioh.problem.BBOB.problems] # list of function names
    d_choice = [10]
    func_select = [(func, d) for func in func_choice for d in d_choice]
    np.random.shuffle(func_select)
    func_i = 0
    instance = 0

    fun, dim = func_select[func_i]
    fun = ioh.get_problem(fun, instance, dim)
    env = DE(fun, 1e4, False, 0.9)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DDQNAgent(state_size, action_size)

    log = []
    for e in range(EPISODES):
        print()
        if func_i == len(func_select): # If all functions have been seen, shuffle
            np.random.shuffle(func_select)
            func_i = 0
            instance += 1
        fun, dim = func_select[func_i]
        func_i += 1

        # Make checkpoints every few episodes (not critical for function)
        if e % 10 == 0:
            agent.save("model3/model/ddqn")
        if e % 100 == 0:
            agent.save("model3/model_"+str(e)+"/ddqn")
            
            # Store buffer (not critical for function)
            name = "model3/buffer_"+str(e)
            if os.path.exists(name+".gz"):
                os.remove(name+".gz")
            fptr = gzip.open(name+".gz", "wb")
            pickle.dump(agent.memory, fptr)
            fptr.close()

            # Save and empty log (not critical for function)
            name = "model3/log_"+str(e)
            if os.path.exists(name+".gz"):
                os.remove(name+".gz")
            fptr = gzip.open(name+".gz", "wb")
            pickle.dump(log, fptr)
            fptr.close()
            log = []

        # Initialise environment with current function
        print(fun,dim)
        fun = ioh.get_problem(fun, instance, dim)
        env = DE(fun, 1e4, True, 0.9)

        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        NP = env.NP

        cumsum = 0
        state = env.reset()
        done = False
        loss_episode = []
        while not done: # repeat until termination
            s = np.reshape(state, [NP, 1, state_size])
            actions = agent.act(s) # select NP actions
            next_state, reward, done, converged = env.step(actions) # do one generation
            agent.memorize(state, actions, reward, next_state, done) # put new experiences in replay buffer
            if not done and converged: # check if termination conditions are met
                b = env.budget
                state = env.reset()
                env.budget = b - env.NP
            else:
                state = next_state
            cumsum += sum(reward)
            if len(agent.memory) > batch_size and instance >= warmup:
                loss = agent.replay(batch_size)
                loss_episode.append(loss)
        print("episode: {}/{}, e: {:.2}".format(e+1, EPISODES, agent.epsilon))
        a = [t[1] for t in agent.memory]
        print(a.count(0), a.count(1), a.count(2), a.count(3), a.count(4), a.count(5), a.count(6), a.count(7))
        
        # append log
        if len(agent.memory) > batch_size and instance >= warmup:
            log.append((cumsum, loss_episode, fun.meta_data.problem_id))
        # refresh target model when necessary
        if e % tau == 0:
            agent._copy_weights()
    
    agent.save("model/model/ddqn")
    
    # Store buffer (not critical for function)
    name = "model/buffer_"+str(EPISODES)
    if os.path.exists(name+".gz"):
        os.remove(name+".gz")
    fptr = gzip.open(name+".gz", "wb")
    pickle.dump(agent.memory, fptr)
    fptr.close()

    # Save log (not critical for function)
    name = "model/log_"+str(EPISODES)
    if os.path.exists(name+".gz"):
        os.remove(name+".gz")
    fptr = gzip.open(name+".gz", "wb")
    pickle.dump(log, fptr)
    fptr.close()