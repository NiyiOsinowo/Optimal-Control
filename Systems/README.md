The goal of this section is to develop a fundamental understanding of what Optimal Control(OC) problems are, their defining characteristics and how to model and represent them. This section will consist of an overview of the core concepts behind optimal control theory and outline it's connection other fields of knowledge. 

To have a better understanding of OC problems I will begin by providing an overview of the core concepts behind optimal control theory including Dynamical Systems, Control Theory and Optimization Problems.

## Table of Contents
1. [Introduction](#1)
2. [Dynamical Systems](#2)
   - [Basic terminology of dynamical systems](#3)
   - [Challenges that may arise when modelling a dynamical system](#4)
   - [Types of dynamical systems](#5)
3. [Control Theory](#6)
   - [Basic terminology of control theory](#7)
   - [Challenges that may arise when controlling a system](#8)
   - [Methods of control](#9)
   - [Markov Decision Processes](#10)
4. [Optimization Problems](#11)
5. [Optimal Control](#12)
6. [Conclusion](#13)

# Introduction <a id ="1"></a>
An Optimal Control problem is fundamentally an optimization problem that involves finding the best possible control strategy for a system to achieve a desired outcome over a specified time horizon. The goal is to find the optimal **control inputs** that minimize or maximize a **performance criterion**, subject to **constraints** on the system's dynamics and the control inputs. 

An optimal control problem can be thought of as a combination of a control problem and optimization problem on a dynamical system.

## Dynamical Systems <a id ="2"></a>
The foundation of an optimal control problem is a dynamical system, which describes how the system's state evolves over time. Dynamical systems are mathematical models that describe how a system's state changes over time. A dynamical system can be modelled in different ways depending on the complexity of the dynamics and other properties of the system but are most commonly represented mathematically by differential equations. The behaviour of dynamical systems are typically expressed as Ordinary Differential Equations(ODEs), which allows us to easily analyse the behaviour of the system by analysing the ODE. 

The dynamics provide the constraints within which the system must operate.

The first step to solving an optimal control problem is system identification(i.e. the process of understanding and modelling the behavior/dynamics of the system). 

### Basic terminology of dynamical systems <a id ="3"></a>

**State Space**: The set of all possible states a dynamical system can occupy, represented mathematically as a vector.

**State Equation**: The differential equation (or difference equation for discrete-time systems) that describes how the system's state evolves over time.

**Equilibrium Points**: States where the system remains stationary, with no further changes in the state vector. A **stable point** is a state $x$ such that for some neighborhood of $x$, the ODE is convergent toward $x$(i.e. $\dot{x} = 0$). A necessary condition for a point to be stable is that it is an *equilibrium point*.

**Stability**: The tendency of a system to remain near or return to an equilibrium point following a perturbation. A dynamic system is said to be Stable( or convergent) for some class of initial states if its solution trajectories do not grow without bound and Unstable(or divergent) otherwise. All stable points are equilibria, but the converse is not true, a point can be an equilibrium without being stable.

**Controllability**: The ability to drive a system from any initial state to any desired final state within a finite time.

**Observability**: The ability to reconstruct the full state of a system from its outputs or observations.

A dynamic system is said to be Autonomous if its dynamics are invariant in time.

### Types of dynamical systems <a id="5"></a>
There are different ways to classify dynamical systems. In this project, we will focus on classifying systems based on how much information about their dynamics is available. This classification is particularly relevant when solving real-world optimal control problems, as the amount of information about the system's dynamics greatly influences the approach to solving the problem.

1. **Deterministic Systems**: 
   If all the factors driving the change of a dynamical system can be identified and modeled precisely, it is called a deterministic system. In these systems, given a set of initial conditions and inputs, the future states can be predicted with certainty.

   Example: A simple pendulum in a vacuum, where the motion is governed by well-defined physical laws.

2. **Stochastic Systems**: 
   If none (or a vanishingly small amount) of the factors driving the change of a dynamical system can be identified with certainty, it is called a stochastic system. These systems involve random elements, and their future states can only be predicted probabilistically.

   Example: The stock market, where numerous unpredictable factors influence price movements.

3. **Partially Known Systems**: 
   In practice, many real-world systems fall between these two extremes. Some aspects of the system may be well-understood and modeled deterministically, while others involve uncertainty or randomness. These systems are often the most challenging and interesting in the context of optimal control.

   Example: A drone flying outdoors, where its basic flight dynamics are known, but wind gusts introduce unpredictable disturbances.

The classification of a system as deterministic, stochastic, or partially known has significant implications for how we approach optimal control problems:

- For deterministic systems, we can often use classical optimal control techniques like the Pontryagin's Maximum Principle or Dynamic Programming.
- Stochastic systems may require approaches from stochastic optimal control theory, such as the Hamilton-Jacobi-Bellman equation for continuous-time systems or Markov Decision Processes for discrete-time systems.
- Partially known systems might benefit from adaptive or robust control strategies that can handle uncertainties in the system model.

Understanding the level of knowledge about a system's dynamics is crucial in choosing appropriate modeling techniques, control strategies, and optimization methods. It also helps in assessing the reliability of the resulting optimal control solutions and in designing control systems that can handle real-world uncertainties and disturbances.

### Challenges that may arise when modelling a dynamical system <a id="4"></a>

When modelling dynamical systems for optimal control, several challenges can arise that complicate the process and affect the accuracy of the model. Understanding these challenges is crucial for developing effective control strategies. We can categorize these challenges into three main areas:

#### 1. System Complexity

a) **Unknown dynamics**: 
   The system's behavior may not be well understood or may be too complex to model accurately. This can occur in systems with intricate interactions or in newly discovered phenomena.

b) **High dimensional systems**:
   Systems with many variables or degrees of freedom can be computationally expensive to model and analyze. The "curse of dimensionality" can make optimization and control problems intractable for high-dimensional systems.

c) **Chaotic/Nonlinear systems**:
   Nonlinear systems can exhibit chaotic behavior, making long-term predictions difficult or impossible. Small changes in initial conditions can lead to vastly different outcomes, complicating control strategies.

d) **Scale differences/changes**: 
   Systems that operate across different scales (e.g., micro and macro scales) can be challenging to model consistently.

#### 2. Observability and Uncertainty

a) **Hidden variables/Partial observability**: 
   Only certain aspects of the state may be measurable by available sensors. For example, a mobile robot with a GPS sensor can only measure position, whereas it may need to model velocity as part of its state. 
   
   Solution approach: State estimation techniques, such as Kalman filtering and particle filtering, can be used to extrapolate the unobserved components of state.

b) **Noise/Disturbances**:
   Real-world systems are often subject to noise and external disturbances. These can be categorized as:
   - Noisy errors: Follow no obvious pattern
   - Systematic errors: Obey a pattern

   These deviations can be further classified as:
   - **Motion uncertainty**: Causes the state to move in unexpected ways at future points in time. 
     Example: Wind gusts moving a drone off its desired path.
   - **State uncertainty**: Occurs when the true state of the system is not known precisely. 
     Example: Measurement errors due to sensor noise.

c) **Modeling error**: 
   This occurs when the modeled dynamics function differs from the actual system dynamics. It can be treated as a form of state uncertainty.

#### 3. Representation and Implementation

a) **Continuous vs. Discrete-time systems**:
   Real-world systems often operate in continuous time, but digital control systems operate at discrete time intervals. Choosing the appropriate representation is crucial.

   - Continuous-time model:
     $\dot{x} = f(x,u) + \epsilon_d$
     Where $\epsilon_d(t) \in E_d$ is some error, and $E_d$ is a set of possible disturbances or a probability distribution over disturbances.

   - Discrete-time model:
     $x_{t+1} = f(x_t, u_t)$
     Here, control changes and state observations occur only at discrete time points.

b) **Open-loop vs. Closed-loop control**:
   - Open-loop systems may "drift" from intended trajectories due to motion uncertainty.
   - Closed-loop controllers can regulate disturbances by choosing controls that drive the system back to the intended trajectory.

#### Addressing the Challenges

1. Use appropriate state estimation techniques for partially observable systems.
2. Employ robust control methods to handle uncertainties and disturbances.
3. Consider adaptive control strategies for systems with unknown or changing dynamics.
4. Use numerical methods and approximations for high-dimensional or complex nonlinear systems.
5. Choose between continuous and discrete-time models based on the specific system and control hardware.
6. Implement closed-loop control when possible to mitigate the effects of uncertainties and disturbances.

Understanding and addressing these challenges is crucial for developing accurate models and effective control strategies in optimal control problems. The choice of modeling approach and control technique often depends on which of these challenges are most prominent in the system under consideration.

## Control Theory <a id ="6"></a>
Control theory is the branch of mathematics that deals with the controlling the behavior of dynamical systems to achieve a desired behavior.

### Basic terminology of control theory <a id ="7"></a>
Control theory introduces several key terms:

- **Control Input**: The variable that can be manipulated to influence the system's behavior.
- **Set Point**: The desired value of the system output.
- **Error**: The difference between the set point and the actual system output.
- **Controller**: The component that determines the control input based on the error.
- **Plant**: The system being controlled.
- **Feedback**: Information about the system's current state used to adjust the control input.
- **Transfer Function**: A mathematical representation of the relationship between the system's input and output.

### Challenges that may arise when controlling a system <a id ="8"></a>
Several challenges can arise when controlling a system:

1. **Time Delays**: Delays between input and observable output can complicate control.
2. **Nonlinearities**: Many real systems have nonlinear behaviors that are difficult to control.
3. **Disturbances**: External factors can interfere with the system's behavior.
4. **Model Uncertainty**: The mathematical model of the system may not perfectly match reality.
5. **Stability Issues**: Ensuring the system remains stable under all conditions can be challenging.
6. **Robustness**: The control system should perform well under various conditions and disturbances.
7. **Actuator Limitations**: Physical constraints on control inputs can limit control effectiveness.

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

## Optimization Problems <a id ="11"></a>
Optimization problems are problems that involve finding the best solution among a set of possible solutions. In the context of dynamical systems, optimization problems often involve finding the control inputs that minimize or maximize a cost function over time.

Key concepts include:

- **Objective Function**: The function to be minimized or maximized.
- **Decision Variables**: The variables that can be adjusted to optimize the objective function.
- **Constraints**: Limitations on the possible values of the decision variables.
- **Feasible Region**: The set of all possible solutions that satisfy the constraints.
- **Optimal Solution**: The solution that gives the best value of the objective function while satisfying all constraints.

Common types of optimization problems include:

1. Linear Programming
2. Nonlinear Programming
3. Integer Programming
4. Dynamic Programming
5. Convex Optimization

## Optimal Control <a id="12"></a>
Optimal control combines elements from dynamical systems, control theory, and optimization to solve problems where the goal is to determine the best possible control strategy for a system over time. Key aspects of optimal control include:

1. **State Equations**: Describe the system dynamics.
2. **Control Inputs**: The variables that can be manipulated to influence the system.
3. **Performance Index (Cost Function)**: The objective function to be minimized or maximized.
4. **Constraints**: Limitations on states and controls.
5. **Boundary Conditions**: Initial and final conditions on the system state.
6. **Time Horizon**: The period over which the control is optimized (finite or infinite).

Optimal control problems can be solved using various methods, including:

- Pontryagin's Maximum Principle
- Dynamic Programming
- Direct Methods (e.g., collocation, shooting methods)
- Indirect Methods (based on necessary conditions for optimality)

Applications of optimal control are widespread, including robotics, aerospace, economics, and process control.

## Conclusion <a id="13"></a>
Understanding optimal control requires a solid grasp of dynamical systems, control theory, and optimization. This interdisciplinary field provides powerful tools for designing and analyzing complex systems to achieve desired performance objectives. As technology advances, optimal control continues to play a crucial role in various fields, from autonomous vehicles to renewable energy systems.

By mastering the concepts outlined in this document, one can approach optimal control problems with a comprehensive understanding of the underlying principles and methodologies. This knowledge forms the foundation for tackling real-world challenges in system design and control.