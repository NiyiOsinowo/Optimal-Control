# Optimal Control Methods

The aim of this project is to explore and compare the different methods of solving optimal control problems.

- Benoit C. CHACHUAT, "Nonlinear and Dynamic Optimization: From Theory to Practice," 
Department of Chemical Engineering, McMaster University, 2009.

The UAV scenario given the paper is shown below.

<img src="https://github.com/cfoh/UAV-Trajectory-Optimization/assets/51439829/27486986-bcb9-47e0-8e8a-8cc2f810f53a" 
  width="400" height="450">

We have modified a certain bit in the environment.

<img width="766" alt="Screenshot 2024-07-30 at 15 57 31" src="https://github.com/user-attachments/assets/ace1325a-b83d-4898-953a-b486b2b91c96">

## The Settings

The settings of the scenario are:
- A 15-by-15 grid map.
- The map contains some obstacles (shown in dark grey).
- The UAV makes a trip of 50 steps from the start cell to the final landing cell.
- The start cell is the left-bottom cell (colored in green), 
  and the final landing cell is also the left-bottom cell (marked by `X`).
- The UAV can only move up/down/left/right in each time step from the center of
  one cell to the center of another cell. It cannot move out of the map 
  nor into the obstacle cells.
- There are three stationary users (or called UEs here) on the map. 
  Their locations are marked as blue circles.
- The UAV communicates simultaneously to the three UEs. Due to the obstacles,
  signals at some cells experience non-line-of-sight (NLOS).
  - Clear cells indicate line-of-sight (LOS) communications with both UEs
  - Light grey cells indicate NLOS from a UE
  - Darker grey cells indicate NLOS from both UEs
  - Dark cells indicate obstacles
- The screenshot shows that the UAV (i.e. the red circle) is in its starting position.

## The Objective

The objective of the design is to propose a learning algorithm such that the UAV makes a 50-step trip from the start cell to the final landing cell while providing good communication service to all 3 UEs. 
The paper has proposed two machine learning (ML) algorithms, namely Q-learning and Deep-Q-Network (DQN), to learn the optimal trajectory.
After sufficient training on the algorithms, the authors observe that:
- the UAV is able to discover an optimal region to serve three UEs
- the UAV reaches the optimal region on a short and efficient path
- the UAV avoids flying through the shadowed areas as the communication
  quality in those areas is not excellent
- after reaching the optimal region, the UAV circles around the 
  region to continue to provide good communication service
- the UAV decides when to start returning back to avoid crashing

## The Code

The code is tested with `Python 3.11.9` with the following packages:
```
Package    Version
---------- -------
keras      3.3.3
numpy      1.26.4
pip        24.0
pygame     2.5.2
setuptools 65.5.0
shapely    2.0.4
pandas     2.2.2
pillow     10.3.0
torch      2.3.1
```

Run the simulation by:
```
uav.py
```

### Neural Network Model

The Proximal Policy Optimization (PPO) algorithm is implemented through a series of key components designed to optimize both policy and value functions using reinforcement learning. The implementation is structured into distinct classes and methods, each serving a specific purpose within the PPO framework.

PPOMemory is a pivotal class in managing and organizing experiences during the training process. It maintains lists of states, actions, probabilities, values, rewards, and done flags. The store_memory method is used to append new experiences to these lists, while the generate_batches method facilitates the creation of mini-batches for training. This batching process helps in efficiently managing memory and ensures that the learning algorithm receives diverse and representative samples. The clear_memory method resets all lists, preparing the memory for new experiences.

```
def store_memory(self, state, action, probs, vals, reward, done):
    self.states.append(state)
    self.actions.append(action)
    self.probs.append(probs)
    self.vals.append(vals)
    self.rewards.append(reward)
    self.dones.append(done)

def generate_batches(self):
    ...
    return self.states, np.array(self.actions), np.array(self.probs), np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches
```

**Actor Network**
The ActorNetwork class defines the policy network responsible for selecting actions. It features a multi-head architecture, which allows it to produce multiple sets of action probabilities. This network consists of shared layers followed by several heads, each producing a probability distribution over possible actions. The forward method computes these distributions, and it optionally applies an action mask to ensure that only valid actions are considered. The network is initialized with orthogonal weights to enhance training stability and convergence. This class also includes methods for saving and loading model checkpoints, which is crucial for preserving and restoring training progress.

```
def forward(self, state, action_mask=None):
    shared_output = self.shared_layers(state)
    head_outputs = [head(shared_output) for head in self.heads]
    if action_mask is not None:
        head_outputs = [dist * action_mask for dist in head_outputs]
        head_outputs = [dist / dist.sum(dim=-1, keepdim=True) for dist in head_outputs]
    distributions = [Categorical(dist) for dist in head_outputs]
    return distributions
```
**Critic Network**

Similarly, the CriticNetwork class is responsible for estimating the value function. It shares a similar architecture to the ActorNetwork, with shared layers followed by heads, though its outputs represent value estimates rather than action probabilities. The forward method generates these value estimates based on the input state. Like the actor network, it includes checkpointing methods to facilitate saving and loading of model weights.

```
def forward(self, state):
    shared_output = self.shared_layers(state)
    values = [head(shared_output) for head in self.heads]
    return values

```
**PPO**
The PPO class integrates these components into the core reinforcement learning algorithm. It initializes the state and action dimensions, as well as hyperparameters such as gamma (the discount factor), policy clipping range, number of epochs for training, entropy coefficient, and GAE lambda. The encode_state method transforms the state into a format suitable for the neural networks. The store_transition method saves the state-action-reward transitions into memory. The execute method handles the decision-making process: it scales the reward, determines if the episode is terminal, converts the state to a tensor, applies the action mask, and samples an action from the policy network. If necessary, it triggers the learning process by calling the learn method.

```
def execute(self, state, reward, is_terminal=False):
    state_np = self.state_to_numpy(state)
    state_tensor = T.tensor(state_np, dtype=T.float).to(self.actor.device)
    action_mask = T.tensor(self.state_action_mask(state), dtype=T.float).to(self.actor.device)
    distributions = self.actor(state_tensor, action_mask)
    dist = distributions[0]
    values = self.critic(state_tensor)
    value = values[0]
    action = dist.sample()
    probs = T.squeeze(dist.log_prob(action)).item()
    action = T.squeeze(action).item()
    value = T.squeeze(value).item()
    self.store_transition(state_np, action, probs, value, reward, is_terminal)
    if self.n_steps % self.learn_after == 0:
        self.learn()
    return action

```
The learn method is the heart of the PPO algorithm, where the actual policy and value function updates occur. It processes the experiences stored in memory to compute advantages, which are then used to update both the actor and critic networks. The advantages are calculated using Generalized Advantage Estimation (GAE), which helps in reducing variance while maintaining bias. During learning, the actor's loss is computed based on the clipped probability ratios to ensure that policy updates remain within a stable range, preventing drastic changes that could destabilize training. The critic’s loss is based on the value function estimates, while an entropy term is included to encourage exploration. The optimizer updates the network parameters to minimize the combined loss, ensuring that the policy and value functions are refined effectively.

```
def learn(self):
    for _ in range(self.n_epochs):
        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()
        advantage = np.zeros(len(reward_arr), dtype=np.float32)
        for t in range(len(reward_arr) - 1):
            ...
            advantage[t] = a_t
        advantage = T.tensor(advantage).to(self.actor.device)
        values = T.tensor(vals_arr).to(self.actor.device)
        for batch in batches:
            ...
            total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coefficient * entropy_loss
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            total_loss.backward()
            self.actor.optimizer.step()
            self.critic.optimizer.step()
    self.memory.clear_memory()
```

Overall, this PPO implementation leverages sophisticated neural network architectures, experience replay, and advanced optimization techniques to address complex reinforcement learning tasks. The structured approach to managing memory, defining neural networks, and updating policies ensures that the agent learns robust and effective strategies in its environment.
### Scenario Parameters

As the paper does not provide full detail of their scenario setup, we make the following assumptions:
- the map is set to 1590-by-1590 in meters
- the UAV is flying at 40 meters above ground
- For the communication, the thermal noise is -174 dBm per Hz

With the above settings, the rate per UE is in the range between around 35 bps/Hz and 50 bps/Hz which is much higher than that of the paper as illustrated in Fig. 3 in the paper. The paper may have used a much wider area than our considered area, and/or the UAV is flying at a much higher altitude.

### Reward Setting

The paper uses transmission rate as the reward for the learning. Depending on the setup, one can use the rate of downlink or uplink transmissions, or both. For uplink transmission, if the transmissions are not orthogonal (in time or frequency), then transmissions of the UEs will interfere with each other. It is unclear which option is used in the paper, here we use orthogonal transmissions and the transmission can be either uplink or downlink. Without loss of generality, we consider uplink transmissions where in our scenario, there are three IoT devices and a UAV. The mission of the UAV is to collect the data from the IoT devices.

Besides, the paper pointed out that the optimal region to serve both UEs is around the shortest midpoint between the two UEs. However, using sum rate as the reward as indicated in (6) in the paper will not create an optimal region at around the shortest midpoint between the two UEs, instead the optimal regions will be around each UE. To match the optimal trajectory shown in the paper, we use minimum rate of both which creates the optimal region around the shortest midpoint between the three UEs. That is:
```python
# reward = r1 + r2   # sum rate, optimal region at around either UE1 or UE2
reward = min(r1,r2,r3)  # min rate, optimal region at the midpoint of UE1 and UE2
```

Apart from using the rate as the reward, we also add additional rewards so that the UAV will return to the final landing cell at the end of its trip. We apply the following rewards:
- If the UAV returns to the final landing cell before the end of its trip,
  we apply penalty to inform the UAV of its premature returning. The penalty is the last 
  immediate reward times the unused time steps. That is, the earlier the UAV returns, the
  more penalty it will receive, so that it will learn to avoid returning earlier. The paper did not apply this penalty, but we found it useful.
- If the UAV fails to return to the final landing cell at the end of its trip, 
  we apply penalty which is the immediate reward times 10. This way, the UAV will learn to return to the final landing cell at the end of its trip to avoid the penalty. This penalty is also described in the paper, although what penalty to apply is not mentioned.

Note that the paper also applies penalty when the UAV moves outside of the map. However, in our design, we simply do not allow the UAV to move outside of the map.

## The Results

<img width="671" alt="image" src="https://github.com/user-attachments/assets/064072f1-de98-451f-8fc4-11f2415e6b39">

The output of PPO (Proximal Policy Optimization) is indeed produced much faster compared to Q-learning and DQN. PPO is designed to balance stability and efficiency, leveraging its architecture to quickly find effective policies.

The ActorNetwork in PPO uses multiple heads to output action probabilities, while the CriticNetwork evaluates state values. The PPO implementation manages exploration and exploitation through the entropy coefficient, which is decayed over time. Despite the stability improvements provided by the clipped objective function and the use of Generalized Advantage Estimation (GAE) for advantage computation, variability in the learned policy can still occur.

Similarly, DQN uses a neural network to approximate Q-values, which can lead to instability and fluctuations around optimal values due to function approximation and experience replay mechanisms.

Despite these fluctuations, both PPO and DQN are effective in complex environments and can converge to near-optimal solutions relatively quickly. PPO is specifically designed to be more stable and sample-efficient than many previous policy gradient methods, though it still experiences some variability due to approximation and the stochastic nature of policy gradient methods.
