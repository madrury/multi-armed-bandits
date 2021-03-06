"""
Simulations of Multi-Armed Bandit learning algorithms.

Following chapter 2 of Reinforcement Learning, by Sutton and Barto.
"""
import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, exp
from numpy.math cimport INFINITY

# Function types for different reward updates and bandit choice strategies.
ctypedef double (*update_rule_t)(long, double, double, double)
ctypedef long (*bandit_choice_rule_t)(long, double[:], long[:], long)

# Constants for indicating learning strategies.
UPDATE_RULE_MEAN = 0
UPDATE_RULE_CONSTANT = 1
BANDIT_CHOICE_RULE_GREEDY = 0
BANDIT_CHOICE_RULE_UCB = 1

"""
Static allocators for bandits.

We take a strategy of a-priori creating all the pulls from a given bandit suite
before applying any learning algorithm, as opposed to pulling the bandits one
at a time during learning.  This has the advantage of requiring only an array
lookup to pull a bandit during learning, instead of generating a standard
normal variate one at a time.  This allows us to use the efficient numpy array
generation methods.
"""
def make_stationary_bandits(n_bandits, n_times, 
                            bandit_grand_mean=0.0,
                            bandit_mean_std=1.0,
                            bandit_std=1.0):
    """Make a suite of stationary bandits.

    The means of these badits are drawn from a standard normal distribution,
    then the actual pulls of the bandtis are standard normally distributed
    around these means.
    
    Parameters
    ----------
    n_bandits: int
      The number of bandits in the suite.

    n_times: int
      The number of pulls of each bandit to generate.

    Returns
    -------
    badtis: np.array of shape (n_bandits, n_pulls)
      The pulls from the suite of badits.
    """
    bandit_means = np.random.normal(
        loc=bandit_grand_mean, scale=bandit_mean_std, size=n_bandits)
    bandit_means = bandit_means.reshape((n_bandits, 1))
    bandit_draws = np.random.normal(scale=bandit_std, size=(n_bandits, n_times))
    return bandit_means + bandit_draws, bandit_means

def non_stationary_bandit_maker(drift=0.01):
    """Make a non-stationary bandit maker function of a specified drift.

    This returns a function which can be called to generate an array of bandit
    pulls of a given non-stationary drift.

    Parameters
    ----------
    drift: float
      The drift of each bandit.

    Returns
    -------
    bandit_maker: function (n_bandits, n_times) -> np.array
      A function that returns an (n_bandits, n_times) array representing the
      pulls from the suite of non-staitonary bandits.
    """
    def make_nonstationary_bandits(n_bandits, n_times):
        bandit_means = np.random.normal(size=n_bandits).reshape((n_bandits, 1))
        bandit_drifts = np.cumsum(
            np.random.normal(scale=drift, size=(n_bandits, n_times)),
            axis=1)
        bandit_non_stationary_means = (
            np.random.normal(size=(n_bandits, n_times)) + bandit_drifts)
        return (bandit_means + bandit_non_stationary_means,
                bandit_non_stationary_means)
    return make_nonstationary_bandits


@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.cdivision(True)
def run_bandits(double[:, :] bandits, 
                double initial_action_estimate=0.0,
                double alpha=1.0,
                double epsilon=0.1, 
                long update_type=UPDATE_RULE_MEAN,
                long bandit_choice_type=BANDIT_CHOICE_RULE_GREEDY,
                **kwargs):
    """
    Run an explore-exploit bandit learning algorithm on a suite of bandits.

    Parameters
    ----------
    badits: np.array of shape (n_bandits, n_times)
      The pulls of the suite of bandits.

    initial_action_estimate: float
      Initial value to use as the estiamte for the reward from the action of
      pulling any bandit in the suite.

    alpha: float
      Step size parameter to use in the constant step size update rule.

    epsilon: float
      Probability of exploring (randomly choosing a bandit) in any given step.

    update_type: int
      The type of rule used to update the action reward estiamtes.

    bandit_choice_type: int
      The type of rule used to choose the best bandit in an exploit step.

    Returns
    -------
    reward_at_stage: np.array of float, shape (n_times,)
      The rewards recieved at every stage of the learning algorithm.

    choice_at_stage: np.array of int, shape (n_times,)
      The bandit chosen at every stage of the learning algorithm.
    """
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
    
    action_estimates = np.full(
        n_bandits, initial_action_estimate, dtype=float)
    n_times_chosen = np.zeros(n_bandits, dtype=int)
    choice_at_stage = np.zeros(n_times, dtype=int)
    reward_at_stage = np.zeros(n_times, dtype=float)
    
    # Lookup tables for random choices made during the learning algorithm.
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
@cython.wraparound(False) 
@cython.cdivision(True)
def run_gradient_bandits(double[:, :] bandits, 
                         double alpha=1.0,
                         double baseline_factor=1.0,
                         **kwargs):
    """Run the gradient bandit learning algroithm on a suite of bandits.

    Parameters
    ----------
    badits: np.array of shape (n_bandits, n_times)
      The pulls of the suite of bandits.

    alpha: float
      Learning rate for the gradient update.

    baseline_factor: float
      Multiplicitive factor to apply to the baseline term in the gradient
      update.  A value of zero turns off the baseline term.  This is used to
      reproduce one of the pictures in barto and sutton.

    Returns
    -------
    reward_at_stage: np.array of float, shape (n_times,)
      The rewards recieved at every stage of the learning algorithm.

    choice_at_stage: np.array of int, shape (n_times,)
      The bandit chosen at every stage of the learning algorithm.
    """
    cdef long n_bandits, n_times, i, j, choice
    cdef double reward, average_reward
    cdef long[:] n_times_chosen, choice_at_stage
    cdef double[:] reward_at_stage, preferences, p, random_unif

    n_bandits = bandits.shape[0]
    n_times = bandits.shape[1]

    n_times_chosen = np.zeros(n_bandits, dtype=int)
    choice_at_stage = np.zeros(n_times, dtype=int)
    reward_at_stage = np.zeros(n_times, dtype=float)
    average_reward = 0.0
    
    preferences = np.zeros(n_bandits, dtype=float)
    p = np.full(n_bandits, 1.0 / n_bandits)
    
    # A lookup table of random numbers used for making random choices according
    # to a discrete probability distribution.
    random_unif = np.random.uniform(size=n_times)
    for i in range(n_times):
        choice = random_choice(n_bandits, p, random_unif[i])
        n_times_chosen[choice] += 1
        reward = bandits[choice, n_times_chosen[choice] - 1]
        # Update average reward with sample mean update rule.
        average_reward += (1.0 / (i + 1)) * (reward - average_reward)
        # Gradient update for preferences.
        for j in range(n_bandits):
            if j != choice:
                preferences[j] -= (
                    alpha * (reward - baseline_factor * average_reward) * p[j])
        preferences[choice] += (
            alpha * (reward - baseline_factor * average_reward) * (1 - p[choice]))
        p = convert_to_probabilities(n_bandits, preferences, p)
        choice_at_stage[i] = choice
        reward_at_stage[i] = reward
    return reward_at_stage, choice_at_stage



# Bandit choice rules.
@cython.boundscheck(False)
cdef long greedy_bandit_choice(long n_bandits,
                               double[:] action_estimates,
                               long[:] n_times_chosen,
                               long step_number):
    """Greedy bandit choice rule.

    This rule always chooses the action with the highest currently estimated
    reward.

    Parameters
    ----------
    n_bandits: int
      The number of bandits in the suite.

    action_estimates: np.array of float, shape (n_bandits,)
      The current reward esttimates for each action.

    n_times_chosen: np.array of int, shape (n_bandits,)
      The number of times each action has been chosen.

    step_number: int
      The total number of steps taken in the learning algorithm.

    Returns
    -------
    choice: int
      The action chosen.
    """
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
    """Upper Confidence Bound selection rule.

    This rule chooses the action with largest currently estiamted upper
    confidence bound.

    Parameters
    ----------
    Same as in other bandit choice rules.

    Returns
    -------
    choice: int
      The action chosen.
    """
    cdef long choice, i
    cdef double best_action_estimate, current_action_estimate
    best_action_estimate = -1 * INFINITY
    choice = 0
    for i in range(n_bandits):
        current_action_estimate = action_estimates[i]
        current_action_estimate += 2.0 * sqrt(log(step_number) / n_times_chosen[i])
        if current_action_estimate > best_action_estimate:
            best_action_estimate = current_action_estimate
            choice = i
    return choice

# Action reward estimate update rules
@cython.cdivision(True)
cdef double sample_average_update(long n, 
                                  double alpha,
                                  double reward,
                                  double action_estimate):
    """Update the reward estimate of an action so that the estimate is always
    the sample mean of all the previously seen rewards.

    Parameters
    ----------
    n: int
      The number of times this action has been chosen.

    alpha: float
      The step size for the update.  Only used in the constant step size update
      rule.

    reward: float
      The reward recieved from this action.

    action_estimate: float
      The currently estimated reward for this action.

    Returns
    -------
    action_update: float
      The update to be made to the current reward estimate for the given
      action.
    """
    return (1.0 / n) * (reward - action_estimate)

cdef double constant_step_update(long n,
                                 double alpha,
                                 double reward,
                                 double action_estimate):
    """Update the reward estiamte of an action using a constant step size
    update rule.

    This ensures that the reward estiamte is always a geometrically weighted
    sum of all the previous rewards.

    Parameters
    ----------
    Same as in other update rules.

    Returns
    -------
    action_update: float
      The update to be made to the current reward estimate for the given
      action.
    """
    return alpha * (reward - action_estimate)

# Gradient Bandit helper functions.
@cython.boundscheck(False)
cdef long random_choice(long n_choices, double[:] p, double runif):
    """Randomly choose from the set {0, 1, ..., n_choices - 1} according to a
    specified probability distribution.

    Parameters
    ----------
    n_choices: int
      The number of choices possible.

    p: np.array of shape (n_choices,)
      A probability distribution over the possible choices.

    ruinf: float
      A random uniform draw.  Used as the "randomness" in this procedure.

    Returns
    -------
    choice: int
      The random choice.
    """
    cdef int i
    cdef double acc = 0
    for i in range(n_choices):
        acc += p[i]
        if acc >= runif:
            return i
    return n_choices - 1

@cython.boundscheck(False)
@cython.cdivision(True)
cdef double[:] convert_to_probabilities(long n_bandits,
                                        double[:] preferences,
                                        double[:] p):
    """Convert an array of preferences to probabilities.

    Parameters
    ----------
    n_bandits: int
      The number of preferences and probabilities.

    preferences: np.array of shape (n_bandits,)
      Preferences.
    
    p: np.array of shape (n_bandits,)
      Array to hold the computed probabilities.  Note that the contents of this
      array will be overwritten.

    Returns
    -------
    p: np.array of shape (n_bandits,)
      Computed probabilities.
    """
    cdef long i
    cdef double s = 0.0
    for i in range(n_bandits):
        p[i] = exp(preferences[i])
        s += p[i]
    for i in range(n_bandits):
        p[i] /= s
    return p
