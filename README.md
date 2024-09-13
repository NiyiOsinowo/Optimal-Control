# Optimal Control Methods

The aim of this project is to explore and compare the different methods of solving optimal control problems.

This project will focus on apply different optimal control methods on different types of systems with the goal.
The optimal control methods implemented in this projects are categorised based on how much information about the system is available as Planning Methods, Model-Based Learning, and Model-Free Learning.
The systems modelled in this project are categorised based on how much information about the system is available as Deterministic systems, and Stochastic Systems.


## The Objective

The objective of this project is to determine the most effective optimal control method for each system 

# Table of Contents

* [Introduction to Optimal Control](#1)
  - [Dynamical Systems](#2)- a brief overview of dynamical systems
    + [Properties of dynamical systems](#3) 
    + [Challenges that may arise when modelling a dynamical system](#4)
    + [Types of dynamical systems](#5)- based on the amount of information needed to model them
  - [Control Theory](#5)
    + [Basic terminology of control theory](#6)
    + [Challenges that may arise when controlling a system](#7)
    + [Methods of control](#8)
    + [Markov Decision Processes](#6)
  - [Optimization Problems](#7)
    + [Types of optimization problems](#8)
    + [Methods of optimization](#9)
    - [Optimization in Control Theory](#10)
* [Optimal Control Methods](#9)
  - [Analytical/ Planning Methods](#10)
    + [Variational Calculus](#11)
    + [Minimum Principle](#12)
    + [Hamilton-Jacobi-Bellman Equation](#13)
  - [Learning Methods](#14)
    + [Dynamic Programming](#15)
    + [Model-Based Reinforcemnt Learning](#16)
    + [Model-Free Reinforment Learning](#17)
* [Implementation of Solutions](#18)
  - [Case Study 1: Trajectory Optimization](#19)
    + [System Design](#20)
    + [Control Implementation](#21)
  - [Case Study 2: Adaptive Control](#22)
    + [System Design](#23)
    + [Control Implementation](#24)
  - [Case Study 3: Stochastic Control](#25)
    + [System Design](#26)
    + [Control Implementation](#27)

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

## Code Design
The code for this project will be written in Python with Jupyter Notebook and will be divided into the following sections:
- **Optimal Control Methods**: This section will contain the implementation of the different optimal control methods.
- **System Models**: This section will contain the implementation of the different system models.
- **Case Studies**: This section will contain the implementation of the different case studies.
- **Tests**: This section will contain the code to test each System Model, and Optimal Control Method
- **Tools**: This section will contain scripts to automate tasks in the project.


The code is tested with `Python 3.12.1` with the following packages:
```
Package    Version
---------- -------
numpy      1.26.4
pip        24.0
torch      2.3.1
```
## References
Model-based Reinforcement Learning: A Survey.(https://arxiv.org/pdf/2006.16712)
