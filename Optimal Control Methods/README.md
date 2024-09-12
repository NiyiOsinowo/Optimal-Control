## Control Theory <a id ="6"></a>
Control theory is the branch of mathematics that deals with the controlling the behavior of dynamical systems to achieve a desired behavior.



### Basic terminology of control theory <a id ="7"></a>

### Challenges that may arise when controlling a system <a id ="8"></a>

### Methods of control <a id ="9"></a>
Feedback/closed-loop control is a control system that adjusts its output based on measured differences between the actual output and the desired output, with the goal of reducing those differences over time.
Open-loop control is a control system that does not adjust its output based on measured differences between the actual output and the desired output.
### Markov Decision Processes <a id ="10"></a>
A Markov Decision Process (MDP) is a mathematical framework for modeling decision-making in situations where
outcomes are partially random and partially under the control of a decision maker. An MDP is a
5-tuple $(S,A,P,R,\gamma)$, where:

- $S$ is a set of states
- $A$ is a set of actions
- $P(s'|s,a)$ is the transition probability from state $s$ to state $
- $R(s,a,s')$ is the reward function
- $\gamma$ is the discount factor

The goal of an MDP is to find a policy $\pi(a|s)$ that maximizes the expected cumulative reward over an infinite horizon.