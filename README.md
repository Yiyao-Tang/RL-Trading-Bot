
# **Project Overview**

### **1.Objective**

This project is a part of my Final Year Project at NTU, and the goal of this project was to design a trading bot which makes long and short order decisions on its own. Deep Q-Learning algorithm was adopted to design the trading bot and the trading bot’s performance was evaluated through backtesting in the SGX market.


## 2. Background

Reinforcement learning (RL) is a branch of machine learning that focuses on training agents to make decisions by maximizing a cumulative reward signal. In RL, an agent interacts with an environment by taking actions, and receives feedback in the form of rewards or penalties based on the outcomes of those actions. Over time, the agent learns to optimize its actions to maximize its cumulative reward.

In the context of trading, reinforcement learning can be used to develop automatic trading bot that are optimized to maximize profit. RL-based trading strategies can learn from past market data by taking actions and receive feedback, which does not require any supervision or labels. It has the ability to adapt to changing market conditions, allowing them to potentially outperform traditional rule-based strategies.
![rl](https://github.com/user-attachments/assets/060db216-711f-437c-bc2e-85e8241eec6d)
As the diagram shows, the RL process starts with an agent in a given state of the environment. The agent selects an action to take based on its current policy, which maps states to actions. The action is then sent to the environment, which transitions to a new state and returns a reward signal to the agent. The agent then updates its policy based on the reward signal and the new state using the Bellman Equation:

V(s)=(R(s,a)+γV(s′))

Where:

s = current state

s’ = next state

a = current action taken

R = return of current state and selected action

V = value of a state

γ = discount factor

In the problem of multiple possible actions, Markov Decision Processes (MDPs) are used to improve Bellman Equation by assigning weightage to the possibility of each action[23]:

V(s)=(R(s,a)+γ∑s′P(s,a,s′)V(s′))

Where:

P(s,a,s’) = Possibility of taking action a in state s and reach next state s’

The MDP process is iterated stochastically through each possible state to update the state value, which sets the foundation for most of the RL algorithms, but different algorithms might adopt different approaches to find the optimal policy which determines the optimal action given a particular state.

In this project, we will focus on Deep Q-learning approach to find the optimal policy and design the trading bot. The designed trading bot would make trading decision on its own and would be back tested to assess its performance.

## 3. Design with Deep Q-Learning Algorithm

In RL algorithms, Q-learning is a popular algorithm used to learn an optimal policy in an MDP (Markov Decision Process) setting. The goal of Q-learning is to learn an optimal action-value function Q(s,a), which represents the expected cumulative reward of taking action a in state s and following the optimal policy thereafter. The Q-value can be updated iteratively using the Bellman equation :

(s,a) ← Q(s,a) + α [r + γ  Q(s',a') - Q(s,a)]

Where:

s = current state

s’ = next state

a = current action taken

r = immediate reward of current state and selected action

V = value of a state

γ = discount factor

α = learning rate

and the optimal policy is defined by:

π*(s) =  Q(s, a)

According to the Q-learning equation, in each step of the action in the training process, the Temporal Difference (TD) Error is calculated as r + γ  Q(s',a') which can be obtained by taking the action a' that maximizes the Q-value in the next state s', and adding the discounted immediate reward r. By subtracting the current value of Q(s,a) and TD Error, we can obtain the discrepancy between the expected reward and the actual reward received: r + γ  Q(s',a') - Q(s,a). This discrepancy would be multiplied by a learning rate α to add to the original Q value and obtain an updated (s,a). This iterated process has been prove to converge to global minimum of loss function and an optimal policy can be defined by selecting the available action with highest Q value.

Traditionally, the training process above is handled by Q-table, a tabular method that stores the Q-values in a table indexed by the state-action pairs. However, the trading environment has high-dimensional(multiple-date data) and continuous state spaces (continuous price value and trading position), whereas Q-table requires discretization of the state space, which can be challenging for high-dimensional and continuous spaces.

Therefore, in this chapter, Deep Q-Network (DQN) algorithms is adopted to design and train the trading agent. DQN uses deep neural networks to approximate the Q-function, which is used to estimate the expected cumulative reward for a given action in a given state.

![Design of trading bot](https://github.com/user-attachments/assets/41297115-3aff-4d42-a922-9f000efac61c)

Based on the structure of DQN model, the trading agent is designed as follows in this chapter:

**State**:

stock’s history close price of the last 90 days

current close price – ordered close price

**Action:** a = argmax(Q(s,a))

Where Q(s,a) = [m, n]

m = value of Q(s, buy)

n = value of Q(s, sell)

**Reward:**

**Policy:** π*(s) =  Q(s, a)

**DNN:** s

Input dimension: 91, Output dimension: 2

**Other Hyperparameters:**

γ (discount factor), α (learning rate), training epoch and hidden layers of DNN would be tuned in the training stage.

## 4.Result
![result](https://github.com/user-attachments/assets/4454026b-d9f3-4545-9473-66ef467305b7)

| Return (Ann.) [%] | Sharpe Ratio | Max. Drawdown [%] |
| --- | --- | --- |
| 22.044 | 1.860 | -4.220 |

The results show that the strategy performed well, with a positive return of 22.04% over the back testing period, which is much higher than the buy and hold return of 3.67%. The Sharpe ratio of 1.86 indicates that the strategy generated satisfactory excess return for the risk taken. Additionally, the Win Rate was 57.89% and the Profit Factor was 1.73, which suggests that the strategy had a good balance of winning and losing trades. Overall, the results indicate that the strategy has potential to be a profitable approach to trading.
