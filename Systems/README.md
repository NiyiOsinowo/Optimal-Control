## Dynamical Systems <a id ="2"></a>
Dynamical systems are mathematical models that describe how a system's state changes over time. Dynamical systems can exhibit a wide range of behaviors, including stability, chaos, and oscillation, depending on the system's parameters and initial conditions. The behaviour of dynamical systems are typically expressed as Ordinary Differential Equations(ODEs), which allows us to easily analyse the behaviour of the system by analysing the ODE. 

In the topic of Dynamical Systems I will cover 
1) Basic Terminology of Dynamical Systems
2) Types of dynamical systems.
3) Challenges that may arise when modelling a dynamical system

### Basic terminology of dynamical systems <a id ="3"></a>

**State Space**: The set of all possible states a dynamical system can occupy, represented mathematically as a vector.

**State Equation**: The differential equation (or difference equation for discrete-time systems) that describes how the system's state evolves over time.

**Equilibrium Points**: States where the system remains stationary, with no further changes in the state vector.

**Stability**: The tendency of a system to remain near or return to an equilibrium point following a perturbation.

**Controllability**: The ability to drive a system from any initial state to any desired final state within a finite time.

**Observability**: The ability to reconstruct the full state of a system from its outputs or observations.

A dynamic system is said to be:

Stable for some class of initial states if its solution trajectories do not grow without bound,

Unstable (or divergent) if the trajectories grow without bound, and

Convergent if the solution trajectories approach a single point.

Autonomous system if its dynamics are invariant in time.

A **stable point** is a state $x$ such that for some neighborhood of $x$, the ODE is convergent toward $x$(i.e. $\dot{x} = 0$). A necessary condition for a point to be stable is that it is an *equilibrium point*.

All stable points are equilibria, but the converse is not true, a point can be an equilibrium without being stable.

### Types of dynamical systems <a id ="5"></a>
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