# Solving Optimal Control Problems <a id="41"></a>

This document explores various methods used to solve optimal control problems, categorized into **Analytical/Planning Methods** and **Learning Methods** based on the amount of information about the system is needed. Both approaches aim to find the control input that optimally minimizes (or maximizes) a performance criterion subject to dynamic constraints.

## Analytical/Planning Methods <a id="42"></a>

Analytical methods provide rigorous mathematical tools for deriving optimal control laws, often using differential equations and variational principles. This method requires full-information about the system to work (i.e. an accurate equation for the dynamics, finite degrees of freedom, knowledge of all forces driving the system).

### Pontryagin's Minimum Principle (PMP) <a id="43"></a>

**Pontryagin's Minimum Principle (PMP)** is a fundamental tool for solving constrained optimal control problems. Given a dynamical system defined by:
$\[
\dot{x}(t) = f(x(t), u(t), t), \quad x(t_0) = x_0
\]$
where \( x(t) \in \mathbb{R}^n \) is the state, and \( u(t) \in \mathbb{R}^m \) is the control input. The objective is to minimize the cost function:
\[
J = \int_{t_0}^{t_f} L(x(t), u(t), t) \, dt + \phi(x(t_f))
\]
subject to the dynamics. PMP introduces the Hamiltonian function:
\[
H(x, u, \lambda, t) = \lambda^\top f(x, u, t) + L(x, u, t)
\]
where \( \lambda(t) \in \mathbb{R}^n \) is the adjoint (or co-state) variable. PMP requires that the optimal control \( u^*(t) \) minimizes the Hamiltonian:
\[
u^*(t) = \arg \min_u H(x^*(t), u, \lambda^*(t), t)
\]
with the adjoint dynamics governed by:
\[
\dot{\lambda}(t) = -\frac{\partial H}{\partial x}(x^*(t), u^*(t), \lambda^*(t), t), \quad \lambda(t_f) = \frac{\partial \phi}{\partial x}(x(t_f)).
\]
PMP offers a set of necessary conditions, but solving these equations often requires numerical methods.

### Variational Calculus <a id="44"></a>

**Variational Calculus** plays a central role in deriving the Euler-Lagrange equations, which form the basis for many optimal control problems. For a functional \( J[x] \) of the form:
\[
J[x] = \int_{t_0}^{t_f} L(x(t), \dot{x}(t), t) \, dt
\]
the problem is to find the function \( x(t) \) that extremizes \( J \). By introducing variations in \( x(t) \), we obtain the Euler-Lagrange equation:
\[
\frac{d}{dt} \left( \frac{\partial L}{\partial \dot{x}} \right) - \frac{\partial L}{\partial x} = 0.
\]
In optimal control, the Lagrangian \( L(x(t), u(t), t) \) depends on both the state and the control input. The resulting Euler-Lagrange equations are closely related to the PMP's Hamiltonian system.

### Minimum Principle <a id="45"></a>

The **Minimum Principle** states that the optimal control minimizes the Hamiltonian function at each time instant:
\[
H(x^*(t), u^*(t), \lambda^*(t), t) \leq H(x^*(t), u, \lambda^*(t), t) \quad \forall u.
\]
For a time-optimal control problem, the minimum principle can be expressed as:
\[
H(x^*(t), u^*(t), \lambda^*(t), t) = 0.
\]
The system's adjoint variables \( \lambda(t) \) provide gradient information for optimizing the control trajectory. The optimal control problem is often transcribed into solving a boundary value problem, where the adjoint system is integrated backward in time.

### Hamilton-Jacobi-Bellman Equation (HJB) <a id="46"></a>

The **Hamilton-Jacobi-Bellman (HJB) equation** is derived from Bellman's principle of optimality. It provides a sufficient condition for optimality in dynamic programming by recursively solving for the value function \( V(x, t) \), which represents the minimal cost-to-go from state \( x \) at time \( t \) to the final state:
\[
V(x, t) = \min_{u(t)} \left[ L(x(t), u(t), t) + V(\dot{x}(t), t+dt) \right].
\]
For small \( dt \), this leads to the HJB equation:
\[
\frac{\partial V(x,t)}{\partial t} + \min_u \left[ \frac{\partial V(x,t)}{\partial x} f(x, u, t) + L(x, u, t) \right] = 0,
\]
with boundary condition:
\[
V(x, t_f) = \phi(x(t_f)).
\]
Solving the HJB equation yields the optimal policy \( u^*(x,t) \). The HJB formulation is powerful but computationally intractable in high-dimensional systems due to the "curse of dimensionality."

## Learning Methods <a id="47"></a>

Learning-based methods, particularly those from reinforcement learning (RL), are effective in settings where an explicit model of the system dynamics is unavailable or when dealing with complex, high-dimensional environments.

### Dynamic Programming <a id="48"></a>

**Dynamic Programming (DP)**, introduced by Bellman, decomposes the control problem into a sequence of simpler subproblems. The key idea is to compute the optimal cost-to-go (value function) \( V(x,t) \) at each state recursively:
\[
V(x,t) = \min_u \left[ L(x, u, t) + V(\dot{x}(t), t+dt) \right].
\]
The optimal control law is then derived from the minimization process. Although DP is conceptually straightforward, solving the full value function in continuous spaces often requires discretization, leading to significant computational challenges.

### Model-Based Reinforcement Learning <a id="49"></a>

**Model-Based Reinforcement Learning (MBRL)** seeks to approximate the environment's dynamics through a learned model \( \hat{f}(x, u) \). The agent then uses this model to simulate future trajectories and optimize the control input. The model can be used to:
1. Plan the control input \( u(t) \) by solving an internal optimization problem.
2. Improve the policy iteratively based on real-time feedback and updates to the model.

In a typical MBRL framework, the agent alternates between learning the model and optimizing the control input using the model. This method offers improved data efficiency compared to model-free approaches, especially in environments with complex but learnable dynamics.

### Model-Free Reinforcement Learning <a id="50"></a>

**Model-Free Reinforcement Learning (MFRL)** directly learns the optimal control policy \( \pi^*(x) \) or the value function \( V(x) \) from interactions with the environment, without constructing an explicit model of the system dynamics. Two popular approaches in MFRL are:

1. **Q-learning**: Learns the action-value function \( Q(x, u) \), which estimates the cumulative reward for taking action \( u \) in state \( x \). The optimal policy is then obtained as:
\[
u^*(x) = \arg \max_u Q(x, u).
\]
The update rule for \( Q \) is based on the Bellman equation:
\[
Q(x,u) \leftarrow Q(x,u) + \alpha \left( r + \gamma \max_{u'} Q(x',u') - Q(x,u) \right),
\]
where \( r \) is the reward, \( \alpha \) is the learning rate, and \( \gamma \) is the discount factor.

2. **Policy Gradient Methods**: Directly optimize the policy \( \pi(u|x) \) by maximizing the expected return \( J(\theta) = \mathbb{E}[R] \) through gradient ascent:
\[
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(u|x) Q^\pi(x,u) \right].
\]
These methods are well-suited for high-dimensional continuous action spaces but often require a large amount of data to converge.

---

## Applications <a id="52"></a>

Optimal control methods find applications in a wide range of fields:
- **Robotics**: Path planning, manipulation, and motion control.
- **Aerospace**: Flight control and trajectory optimization for spacecraft and aircraft.
- **Economics**: Optimal resource allocation and decision-making under uncertainty.
- **Biomedicine**: Optimal drug dosing and treatment planning.

## Future Work <a id="53"></a>

1. **Combining Analytical and Learning Methods**: Explore hybrid approaches that integrate PMP with reinforcement learning for complex, real-world problems.
2. **Real-time Control**: Develop faster solvers for online applications in robotics and autonomous systems.
3. **Efficient Exploration**: Improve reinforcement learning methods to enhance exploration in high-dimensional and continuous environments.

## References <a id="54"></a

>

- Pontryagin, L. S., et al. *The Mathematical Theory of Optimal Processes*. 1962.
- Bellman, R. *Dynamic Programming*. 1957.
- Sutton, R. S., Barto, A. G. *Reinforcement Learning: An Introduction*. 2018.

---

This version expands the mathematical descriptions for each method, providing more in-depth explanations of the principles and equations behind them.