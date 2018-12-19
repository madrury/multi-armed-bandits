import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log
from numpy.math cimport INFINITY

ctypedef double (*update_rule_t)(long, double, double, double)
ctypedef long (*bandit_choice_rule_t)(long, double[:], long[:], long)

UPDATE_RULE_MEAN = 0
UPDATE_RULE_CONSTANT = 1

BANDIT_CHOICE_RULE_GREEDY = 0
BANDIT_CHOICE_RULE_UCB = 1


def make_stationary_bandits(n_bandits, n_times):
    bandit_means = np.random.normal(size=n_bandits).reshape((n_bandits, 1))
    bandit_draws = np.random.normal(size=(n_bandits, n_times))
    return bandit_means + bandit_draws, bandit_means

def non_stationary_bandit_maker(drift=0.01):
    def make_nonstationary_bandits(n_bandits, n_times):
        bandit_means = np.random.normal(size=n_bandits).reshape((n_bandits, 1))
        bandit_drifts = np.cumsum(
            np.random.normal(scale=drift, size=(n_bandits, n_times)),
            axis=1)
        bandit_draws = np.random.normal(size=(n_bandits, n_times)) + bandit_drifts
        return bandit_means + bandit_draws
    return make_nonstationary_bandits


@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.cdivision(True)
def run_bandits(double[:, :] bandits, 
                double alpha=1.0,
                double epsilon=0.1, 
                long update_type=UPDATE_RULE_MEAN,
                long bandit_choice_type=BANDIT_CHOICE_RULE_GREEDY):
    cdef long n_bandits, n_times, i, greedy, choice
    cdef double reward
    cdef long[:] take_greedy, random_choices, n_times_chosen, choice_at_stage
    cdef double[:] action_estimates, reward_at_stage
    cdef update_rule_t update_rule
    cdef bandit_choice_rule_t bandit_choice_rule
    
    n_bandits = bandits.shape[0]
    n_times = bandits.shape[1]
    
    if update_type == UPDATE_RULE_MEAN:
        update_rule = sample_average_update
        alpha = 1.0
    elif update_type == UPDATE_RULE_CONSTANT:
        update_rule = constant_step_update

    if bandit_choice_type == BANDIT_CHOICE_RULE_GREEDY:
        bandit_choice_rule = greedy_bandit_choice
    elif bandit_choice_type == BANDIT_CHOICE_RULE_UCB:
        bandit_choice_rule = ucb_bandit_choice
    
    action_estimates = np.zeros(n_bandits, dtype=float)
    n_times_chosen = np.zeros(n_bandits, dtype=int)
    choice_at_stage = np.zeros(n_times, dtype=int)
    reward_at_stage = np.zeros(n_times, dtype=float)
    
    take_greedy = 1 - np.random.binomial(1, p=epsilon, size=n_times)
    random_choices = np.random.choice(n_bandits, size=n_times)
    for i in range(n_times):
        greedy = take_greedy[i]
        if greedy:
            choice = bandit_choice_rule(
                n_bandits, action_estimates, n_times_chosen, i)
        else:
            choice = random_choices[i]
        n_times_chosen[choice] += 1
        reward = bandits[choice, n_times_chosen[choice] - 1]
        action_estimates[choice] += update_rule(
            n_times_chosen[choice], alpha, reward, action_estimates[choice])
        reward_at_stage[i] = reward
        choice_at_stage[i] = choice
    return reward_at_stage, choice_at_stage


@cython.boundscheck(False)
cdef long greedy_bandit_choice(long n_bandits,
                               double[:] action_estimates,
                               long[:] n_times_chosen,
                               long step_number):
    cdef long choice, i
    cdef double best_action_estimate
    best_action_estimate = -1 * INFINITY
    choice = 0
    for i in range(n_bandits):
        if action_estimates[i] > best_action_estimate:
            best_action_estimate = action_estimates[i]
            choice = i
    return choice

@cython.boundscheck(False)
@cython.cdivision(True)
cdef long ucb_bandit_choice(long n_bandits,
                            double[:] action_estimates,
                            long[:] n_times_chosen,
                            long step_number):
    cdef long choice, i
    cdef double best_action_estimate, current_action_estimate
    best_action_estimate = -1 * INFINITY
    choice = 0
    for i in range(n_bandits):
        current_action_estimate = action_estimates[i]
        current_action_estimate += 5 * sqrt(log(step_number) / n_times_chosen[i])
        if current_action_estimate > best_action_estimate:
            best_action_estimate = current_action_estimate
            choice = i
    return choice

@cython.cdivision(True)
cdef double sample_average_update(long n, 
                                  double alpha,
                                  double reward,
                                  double action_estimate):
    return (1/n) * (reward - action_estimate)

cdef double constant_step_update(long n,
                                 double alpha,
                                 double reward,
                                 double action_estimate):
    return alpha * (reward - action_estimate)
