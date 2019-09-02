import numpy as np
import random
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = 0.0005
        self.gamma = 1.0
        self.alpha = 0.01

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        #return np.random.choice(self.nA)
        # Select greedy action with probability epsilon
        if random.random() > self.eps:
            return np.argmax(self.Q[state])
        # Otherwise, select an action randomly
        else:
            return random.choice(np.arange(self.nA))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # Q-table estimate for current state, action pair
        current = self.Q[state][action]
        ####### Q-SARSA ###########
        # Next State value
        #Qsa_next = np.max(self.Q[next_state]) if next_state is not None else 0
        
        ###### Expected SARSA ########
        # Current policy for next state S`
        policy_s = np.ones(self.nA) * self.eps / self.nA
        # Greedy Action
        policy_s[np.argmax(self.Q[next_state])] = 1 - self.eps + (self.eps / self.nA)
        # State value at next time step
        Qsa_next = np.dot(self.Q[next_state], policy_s)
        # Construct TD target
        target = reward + (self.gamma * Qsa_next)
        # Get updated value
        new_value = current + (self.alpha * (target - current))
        self.Q[state][action] = new_value