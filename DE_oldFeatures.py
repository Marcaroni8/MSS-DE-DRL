from __future__ import division
import numpy as np
from numpy.random import rand
import gym
from gym import spaces
from scipy.spatial import distance
from scipy.stats import rankdata

### mutation algorithms ###
def rand1(population, samples, scale, best, i): # DE/rand/1
    r0, r1, r2 = samples[:3]
    return (population[r0] + scale * (population[r1] - population[r2]))

def rand2(population, samples, scale, best, i): # DE/rand/2
    r0, r1, r2, r3, r4 = samples[:5]
    return (population[r0] + scale * (population[r1] - population[r2] + population[r3] - population[r4]))

def rand_to_best2(population, samples, scale, best, i): # DE/rand-to-best/2
    r0, r1, r2, r3, r4 = samples[:5]
    return (population[r0] + scale * (population[best] - population[r0] + population[r1] - population[r2] + population[r3] - population[r4]))

def current_to_rand1(population, samples, scale, best, i): # DE/current-to-rand/1
    r0, r1, r2 = samples[:3]
    return (population[i] + scale * (population[r0] - population[i] + population[r1] - population[r2]))


### state function helpers ###
def select_samples(popsize, number_samples):
    """
    obtain random integers from range(popsize),
    without replacement.  You can't have the original candidate either.
    """
    r = np.zeros((popsize, number_samples), dtype=int)
    for i in range(popsize):
        idxs = list(range(popsize))
        idxs.remove(i)
        r[i] = np.random.choice(idxs, number_samples, replace=False)
    return r

def normalise(a, mi, mx):
    a = (a - mi) / (mx - mi)
    return a

def count_success(popsize, gen_window, i, j, Off_met):
    c_s = 0; c_us = 0
    c_s = np.sum((gen_window[j, :, 0] == i) & (gen_window[j, :, Off_met] != -1))
    c_us = np.sum((gen_window[j, :, 0] == i) & (gen_window[j, :, Off_met] == -1))
    return c_s, c_us

                                                        ##########################Success based###########################################
# Applicable for fix number of generations
def Success_Rate1(popsize, n_ops, gen_window, Off_met, max_gen):
    state_value = np.zeros(n_ops)
    gen_window = np.array(gen_window)
    if len(gen_window) < max_gen:
        max_gen = len(gen_window)
    for i in range(n_ops):
        appl = 0; t_s = 0
        for j in range(len(gen_window)-1, len(gen_window)-max_gen-1, -1):
            total_success = 0; total_unsuccess = 0
            if np.any(gen_window[j, :, 0] == i):
                total_success, total_unsuccess = count_success(popsize, gen_window, i, j, Off_met)
                t_s += total_success
                appl += total_success + total_unsuccess
        if appl != 0:
            state_value[i] = t_s / appl
    return state_value

                                                        ##########################Weighted offspring based################################
# Applicable for fix number of generations
def Weighted_Offspring1(popsize, n_ops, gen_window, Off_met, max_gen):
    state_value = np.zeros(n_ops)
    gen_window = np.array(gen_window)
    if len(gen_window) < max_gen:
        max_gen = len(gen_window)
    for i in range(n_ops):
        appl = 0
        for j in range(len(gen_window)-1, len(gen_window)-max_gen-1, -1):
            total_success = 0; total_unsuccess = 0
            if np.any(gen_window[j, :, 0] == i):
                total_success, total_unsuccess = count_success(popsize, gen_window, i, j, Off_met)
                state_value[i] += np.sum(gen_window[j, np.where((gen_window[j, :, 0] == i) & (gen_window[j, :, Off_met] != -1)), Off_met])
                appl += total_success + total_unsuccess
        if appl != 0:
            state_value[i] = state_value[i] / appl
    if np.sum(state_value) != 0:
        state_value = state_value / np.sum(state_value)
    return state_value

                                                        ##########################Best offspring based#############################
# Applicable for fix number of generations
def Best_Offspring1(popsize, n_ops, gen_window, Off_met, max_gen):
    state_value = np.zeros(n_ops)
    gen_window = np.array(gen_window)
    best_t = np.zeros(n_ops); best_t_1 = np.zeros(n_ops)
    for i in range(n_ops):
        # for last 2 generations
        n_applications = np.zeros(2)
        # Calculating best in current generation
        if np.any((gen_window[len(gen_window)-1, :, 0] == i) & (gen_window[len(gen_window)-1, :, Off_met] != -1)):
            total_success, total_unsuccess = count_success(popsize, gen_window, i, len(gen_window)-1, Off_met)
            n_applications[0] = total_success + total_unsuccess
            best_t[i] = np.max(gen_window[len(gen_window)-1, np.where((gen_window[len(gen_window)-1, :, 0] == i) & (gen_window[len(gen_window)-1, :, Off_met] != -1)), Off_met])
        # Calculating best in last generation
        if len(gen_window)>=2 and np.any((gen_window[len(gen_window)-2,:,0] == i) & (gen_window[len(gen_window)-2, :, Off_met] != -1)):
            total_success, total_unsuccess = count_success(popsize, gen_window, i, len(gen_window)-2, Off_met)
            n_applications[1] = total_success + total_unsuccess
            best_t_1[i] = np.max(gen_window[len(gen_window)-2, np.where((gen_window[len(gen_window)-2, :, 0] == i) & (gen_window[len(gen_window)-2, :, Off_met] != -1)), Off_met])
        if best_t_1[i] != 0 and np.fabs(n_applications[0] - n_applications[1]) != 0:
            state_value[i] = np.fabs(best_t[i] - best_t_1[i]) / ((best_t_1[i]) * (np.fabs(n_applications[0] - n_applications[1])))
        elif best_t_1[i] != 0 and np.fabs(n_applications[0] - n_applications[1]) == 0:
            state_value[i] = np.fabs(best_t[i] - best_t_1[i]) / (best_t_1[i])
        elif best_t_1[i] == 0 and np.fabs(n_applications[0] - n_applications[1]) != 0:
            state_value[i] = np.fabs(best_t[i] - best_t_1[i]) / (np.fabs(n_applications[0] - n_applications[1]))
        else:
            state_value[i] = np.fabs(best_t[i] - best_t_1[i])
    if np.sum(state_value) != 0:
        state_value = state_value / np.sum(state_value)
    return state_value

# Applicable for fix number of generations
def Best_Offspring2(popsize, n_ops, gen_window, Off_met, max_gen):
    state_value = np.zeros(n_ops)
    gen_window = np.array(gen_window)
    if len(gen_window) < max_gen:
        max_gen = len(gen_window)
    for i in range(n_ops):
        gen_best = []
        for j in range(len(gen_window)-1, len(gen_window)-max_gen-1, -1):
            if np.any((gen_window[j,:,0] == i) & (gen_window[j, :, Off_met] != -1)):
                gen_best.append(np.max(np.hstack(gen_window[j, np.where((gen_window[j,:,0] == i) & (gen_window[j, :, Off_met] != -1)), Off_met])))
                state_value[i] += np.sum(gen_best)
    if np.sum(state_value) != 0:
        state_value = state_value / np.sum(state_value)
    return state_value


class DE(gym.Env):
    def __init__(self, fun, max_budget:int=1e4, state_features:bool=True, CR:float=1.0):
        # set the mutation strategies
        self.mu_ops = [rand1, rand2, rand_to_best2, current_to_rand1]
        self.mu_FF = [0.3,0.8]
        self.mutations = [[m, f] for m in self.mu_ops for f in self.mu_FF]

        self.n_ops = len(self.mutations)
        self.action_space = spaces.Discrete(self.n_ops)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(18+16*self.n_ops,), dtype = np.float32)
        
        self.NP = 100
        self.max_gen = 10
        self.number_metric = 5
        self.fun = fun
        self.lbounds = fun.bounds.lb
        self.ubounds = fun.bounds.ub
        self.dim = fun.meta_data.n_variables
        self.best_value = fun.optimum.y
        self.max_budget = max_budget
        self.CR = CR # crossover-rate
        self.state_features = state_features

    def step(self, actions = []): # each step is one generation
        """Mutate, Crossover and Evaluation, and compute Reward and State"""
        if len(actions) == 0:
            self.u = np.copy(self.X)
        else:
            self.opu = actions
            m = [self.mutations[i] for i in actions]
        
            # mutation
            self.r = select_samples(self.NP, 5)
            v = np.zeros((self.NP, self.dim))
            for i in range(self.NP):
                v[i] = m[i][0](self.X, self.r[i], m[i][1], self.best, i)

            # bounds correction
            v = np.clip(v, self.lbounds[0], self.ubounds[0])

            if self.CR == 1.:
                self.u = np.copy(v)
            else:
                # select parents for crossover
                self.crossovers = np.random.rand(self.NP, self.dim) < self.CR
                # for i in range(self.NP):
                self.fill_points = np.random.randint(self.dim, size = self.NP)
                self.crossovers[np.arange(self.NP), self.fill_points] = True
                # crossover (as described by Storn and Price)
                self.u = np.where(self.crossovers, v, self.X)
            
            # function evaluation
            self.F1 = [self.fun(i) for i in self.u]
            self.budget -= self.NP
        

        reward = np.zeros(self.NP,dtype=int)
        generation_metric = np.zeros((self.NP, self.number_metric))
        generation_metric[:,0] = self.opu if len(actions) != 0 else -np.ones(self.NP) # first time do just -1 TODO find better solution for Best_Offspring2() (wat was dit ookalweer?)
        
        for i in range(self.NP):
            if self.F1[i] <= self.F[i]:
                generation_metric[i,1] = self.F[i] - self.F1[i]
                if self.F1[i] < np.min(self.F):
                    generation_metric[i,2] = np.min(self.F) - self.F1[i]
                else:
                    generation_metric[i,2] = -1

                if self.F1[i] < self.best_so_far:
                    generation_metric[i,3] = self.best_so_far - self.F1[i]
                    self.best_so_far = self.F1[i]
                    self.best_so_far_position = self.u[i]
                    self.stagnation_count = 0
                    reward[i] = 10
                else:
                    generation_metric[i,3] = -1
                    reward[i] = 1
                    self.stagnation_count += 1

                if self.F1[i] < np.median(self.F):
                    generation_metric[i,4] = np.median(self.F) - self.F1[i]
                else:
                    generation_metric[i,4] = -1

                if self.worst_so_far < self.F1[i]:
                    self.worst_so_far = self.F1[i]
                
                self.F[i] = self.F1[i]
                self.X[i] = self.u[i]
                # if generation_metric[i,2] != generation_metric[i,3]:
                #     print("gm")
                #     print(generation_metric[i,2] - generation_metric[i,3])
                # print(generation_metric[i,2] - generation_metric[i,3])
                # print(generation_metric[i,2])
            else:
                generation_metric[i,1:self.number_metric] = [-1] * 4
        
        
        self.gen_window.append(generation_metric)

        ob = np.zeros(18+16*self.n_ops)
        if self.state_features:
            copy_ob = np.empty(0)
            copy_ob = np.concatenate((copy_ob, Success_Rate1(self.NP, self.n_ops, self.gen_window, 1, self.max_gen)))
            copy_ob = np.concatenate((copy_ob, Success_Rate1(self.NP, self.n_ops, self.gen_window, 2, self.max_gen)))
            copy_ob = np.concatenate((copy_ob, Success_Rate1(self.NP, self.n_ops, self.gen_window, 3, self.max_gen)))
            copy_ob = np.concatenate((copy_ob, Success_Rate1(self.NP, self.n_ops, self.gen_window, 4, self.max_gen)))
            
            copy_ob = np.concatenate((copy_ob, Weighted_Offspring1(self.NP, self.n_ops, self.gen_window, 1, self.max_gen)))
            copy_ob = np.concatenate((copy_ob, Weighted_Offspring1(self.NP, self.n_ops, self.gen_window, 2, self.max_gen)))
            copy_ob = np.concatenate((copy_ob, Weighted_Offspring1(self.NP, self.n_ops, self.gen_window, 3, self.max_gen)))
            copy_ob = np.concatenate((copy_ob, Weighted_Offspring1(self.NP, self.n_ops, self.gen_window, 4, self.max_gen)))
            
            copy_ob = np.concatenate((copy_ob, Best_Offspring1(self.NP, self.n_ops, self.gen_window, 1, self.max_gen)))
            copy_ob = np.concatenate((copy_ob, Best_Offspring1(self.NP, self.n_ops, self.gen_window, 2, self.max_gen)))
            copy_ob = np.concatenate((copy_ob, Best_Offspring1(self.NP, self.n_ops, self.gen_window, 3, self.max_gen)))
            copy_ob = np.concatenate((copy_ob, Best_Offspring1(self.NP, self.n_ops, self.gen_window, 4, self.max_gen)))
            
            copy_ob = np.concatenate((copy_ob, Best_Offspring2(self.NP, self.n_ops, self.gen_window, 1, self.max_gen)))
            copy_ob = np.concatenate((copy_ob, Best_Offspring2(self.NP, self.n_ops, self.gen_window, 2, self.max_gen)))
            copy_ob = np.concatenate((copy_ob, Best_Offspring2(self.NP, self.n_ops, self.gen_window, 3, self.max_gen)))
            copy_ob = np.concatenate((copy_ob, Best_Offspring2(self.NP, self.n_ops, self.gen_window, 4, self.max_gen)))

            ob[18:18+16*self.n_ops] = copy_ob
            
            self.generation = self.generation + 1
            self.best = np.argmin(self.F)
            self.pop_average = np.average(self.F)
            self.pop_std = np.std(self.F)
            self.max_std = np.std((np.repeat(self.best_so_far, self.NP/2), np.repeat(self.worst_so_far, self.NP/2)))

            # Population fitness statistic
            ob[1] = normalise(self.pop_average, self.best_so_far, self.worst_so_far)
            ob[2] = self.pop_std / self.max_std
            ob[3] = self.budget / self.max_budget
            ob[4] = self.stagnation_count / self.max_budget
            
            ob = np.tile(ob, (self.NP,1))

            self.r = select_samples(self.NP, 5)
            for i in range(self.NP):
                # Parent fitness
                ob[i,0] = normalise(self.F[i], self.best_so_far, self.worst_so_far)
                # Random sample based observations
                ob[i,5:11] = distance.cdist(self.X[[self.r[i,0],self.r[i,1],self.r[i,2],self.r[i,3],self.r[i,4],self.best]], np.expand_dims(self.X[i], axis=0)).T / self.max_dist
                ob[i,11:17] = np.fabs(self.F[[self.r[i,0],self.r[i,1],self.r[i,2],self.r[i,3],self.r[i,4],self.best]] - self.F[i]) / (self.worst_so_far - self.best_so_far)
                ob[i,17] = distance.euclidean(self.best_so_far_position, self.X[i]) / self.max_dist
                # if ob[i,0] != ob[i,16]:
                #     print("0,16")
                #     print(ob[i,0] - ob[i,16])
                # if ob[i,10] != ob[i,17]:
                #     print("10,17")
                #     print(ob[i,10] - ob[i,17])
        else:
            ob = np.tile(ob, (self.NP,1))

        if self.budget <= 0 or self.best_so_far - 1e-8 <= self.best_value:
            print("$$$", self.budget, self.best_value, self.best_so_far,"$$$")
            return ob, reward, True, max(self.F) - min(self.F) < 1e-9
        else:
            return ob, reward, False, max(self.F) - min(self.F) < 1e-9

    def reset(self):
        """Initialize the population and do the first evaluation step."""
        self.budget = self.max_budget
        self.generation = 0
        self.X = self.lbounds + ((self.ubounds - self.lbounds) * np.random.rand(self.NP, self.dim))
        self.F = np.array([self.fun(x) for x in self.X])
        self.u = np.zeros((self.NP, self.dim))
        self.F1 = np.copy(self.F)
        self.budget -= self.NP
    
        self.best_so_far = np.min(self.F)
        self.best_so_far_position = self.X[np.argmin(self.F)]
        self.worst_so_far = np.max(self.F)

        self.best = np.argmin(self.F)
        
        self.gen_window = [] # shape: (generations, NP, number_metric) or (generations, 100, 5)
        
        self.max_dist = distance.euclidean(self.lbounds, self.ubounds)
        self.stagnation_count = 0
        return self.step()[0]