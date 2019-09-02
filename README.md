# DRLND
Deep Reinforcement Learning Nanodegree Mini Projects

# 1) Monte Carlo Methods Code
Implement two of the Monte Carlo (MC) algorithms.

### Part 1: MC Prediction
Implement MC prediction (for estimating the action-value function).

### Part 2: MC Control
Implement constant-alpha MC control.

# 2) Temporal-Difference Methods Code
Implementations of three Temporal-Difference (TD) methods:
- SARSA
- SARSA-Max (Q-Learning)
- Expected SARSA

The algorithms were tested with the [CliffWalking OpenAI Gym environment](https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py).

# 3) OpenAI Gym’s Taxi-v2 Task
Design an algorithm to teach a taxi agent to navigate a small gridworld using OpenAI Gym's [Taxi-v2](https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py) environment.

Best average reward achieved = **9.3947**, using:
- **Expected SARSA** method
- Alpha = 0.01
- Epsilon = 0.0005
- Gamma = 1.0
- 50,000 episodes
