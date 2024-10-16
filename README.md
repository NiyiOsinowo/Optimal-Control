# Optimal Control

The aim of this project is to explore and compare the different methods of solving optimal control problems.

This project will focus on apply different optimal control methods on different types of systems with the goal.

The optimal control methods implemented in this projects are categorised based on how much information about the system is available as Planning Methods which require complete information on the system, Model-Based Learning which require some information on the system, and Model-Free Learning which requires almost no information on the system.

The systems modelled in this project are also categorised based on how much information about the system is available as Deterministic systems, and Stochastic Systems.

## The Objective

The objective of this project is to implement an optimal control strategy for different classes of optimal control problems

# Table of Contents

* [Introduction to Optimal Control Problems](./Systems/README.md)
  - [Dynamical Systems](./Systems/README.md)
    + [Properties of dynamical systems](./Systems/README.md) 
    + [Challenges that may arise when modelling a dynamical system](./Systems/README.md)
    + [Types of dynamical systems](./Systems/README.md)
  - [Control Theory](./Systems/README.md)
    + [Basic terminology of control theory](./Systems/README.md)
    + [Challenges that may arise when controlling a system](./Systems/README.md)
    + [Methods of control](./Systems/README.md)
    + [Markov Decision Processes](./Systems/README.md)
  - [Optimization Problems](./Systems/README.md)
    + [Types of optimization problems](./Systems/README.md)
    + [Methods of optimization](./Systems/README.md)
    - [Optimization in Control Theory](./Systems/README.md)
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
