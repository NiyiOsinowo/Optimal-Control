The goal of this section is to develop a fundamental understanding of what Optimal Control(OC) problems are, their defining characteristics and how to model and represent them. This section will consist of an overview of the core concepts behind optimal control theory and outline it's connection other fields of knowledge. 

To have a better understanding of OC problems I will begin by providing an overview of the core concepts behind optimal control theory including Dynamical Systems, Control Theory and Optimization Problems.

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

### Types of dynamical systems <a id ="5"></a>
Many systems fall under the There are different ways to classify dynamical systems, in this project I will focus on classifying systems based on how much information about its dynamcis is available. In this project I have chosen to focus on a more practical feature when it comes to solving real world optimal control problems and that is the amount of information about the dynamics of the system that is available.

If all the factors driving the change of a dynamical system can be identified, it is called a deterministic system. If none(or lim->0) the factors driving the change of a dynamical system can be identified, it is called a stochastic system.

### Challenges that may arise when modelling a dynamical system <a id ="4"></a>

- Challenges that may arise when modelling a dynamical system:
    
    - **Unknown dynamics**: The system's dynamics are not well understood

    - **High dimensional systems**:
    
    - **Chaotic/Nonlinear systems**:
    
    - **Hidden variables/ Partial observability**: Partial observability means that only certain aspects of the state can possibly be measured by the available sensors. For example, a mobile robot with a GPS sensor can only measure position, whereas it may need to model velocity as part of its state. State estimation techniques, such as Kalman filtering and particle filtering, can be used to extrapolate the unobserved components of state to provide reasonable state estimates. With those estimates, there will be some remaining localization error that the controller will still need to handle.
    
    - **Noise/ Disturbances**
    
    - **Scale differences/ changes**: Generally speaking, errors can be characterized as being either noisy or systematic. A noisy error is one obeys no obvious pattern each time it is measured. A systematic error is one that does obey a pattern. These deviations fall can also be categorized as **motion uncertainty** and **state uncertainty**. Disturbances are a form of motion uncertainty that cause the state to be moved in unexpected ways at future points in time. For example, wind gusts are very hard to predict in advance, and can move a drone from a desired path. Actuation error occurs when a desired control is not executed faithfully. These errors can be treated as motion uncertainty. Measurement error is a type of state uncertainty where due to sensor noise the state is observed incorrectly. Understanding measurement error is critical for closed-loop controllers which base their behavior on the measured state. Modeling error, means that the true dynamics function differs from the actual dynamics of the system. This is sometimes considered a third class of uncertainty, but could also be treated as state uncertainty.

Motion uncertainty can be modeled as a disturbance to the dynamics
\dot{x} = f(x,u) + \epsilon_d
 where \epsilon_{d}(t) in $E_{d}$ is some
error. Here $E_d$ is a set of possible disturbances, or a probability
distribution over disturbances. Motion uncertainty will cause an
open-loop system to "drift" from its intended trajectory over time. A
properly designed closed-loop controller can regulate the disturbances
by choosing controls that drive the system back to intended trajectory.

In many cases it is convenient to talk about discrete-time systems in which time is no longer a continuous variable but a discrete quantity  t=0,1,2,…, and the dynamics are specified in the form

$x_{t+1}=f(x_{t},u_{t})$

Here, the control is allowed to change only at discrete points in time, and the state is only observed at discrete points in time. This more accurately characterizes digital control systems which operate on a given clock frequency. However, in many situations the control frequency is so high that the continuous-time model (2) is appropriate.

Usually systems of the form

$\ddot{x}=f(x,\dot{x},t)$

which relate state and controls to accelerations of the state  $\ddot{x}=\frac{dx^2}{d^2x}$
 . This does not seem to satisfy our definition of a dynamic system, since we've never seen a double time derivative. However, we can employ a stacking trick to define a first order system, but of twice the dimension. Let us define the stacked state vector

  \begin{align}
    y &= \begin{pmatrix}
           x \\
           \dot{x}
         \end{pmatrix}
  \end{align}

Then, we can rewrite ( 4 ) in a first-order form as:

$\dot{y}=g(y, u)$(6)
where  $g(y, u)=f(x, \dot{x}, u)$ simply "unstacks" the state and velocity from  $y$ . Now all of the machinery of first-order systems can be applied to the second order system. This can also be done for dynamic systems of order 3 and higher, wherein all derivatives are stacked into a single vector.

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

## Optimization Problems <a id ="11"></a>
Optimization problems are problems that involve finding the best solution among a set of possible solutions. In the
context of dynamical systems, optimization problems often involve finding the control inputs that minimize or
maximize a cost function over time.

