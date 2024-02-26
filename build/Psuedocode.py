# An agent trying to get the most reward by traversing a network
from numpy import *
from matplotlib import *
from dataclasses import *
import random
from unittest import *

class State(object):
    def __init__(self, Parameters):
        self.Value= None
        self.neighbors = {}
        self.__Parameter = Parameters
    @property
    def Parameter(self):
        return self.__Parameter

    def add_neighbor(self, new_neighbor):
        if isinstance(new_neighbor, State) ==False:
            raise Exception("Invalid neighbor Type")
        elif new_neighbor in self.neighbors.values():
            raise Exception("Neighbor Duplicate")
        else: 
            self.neighbors["Neighbor{0}".format(len(self.neighbors))]= new_neighbor

    def __str__(self) -> str:
        State_Info= '{0}'
        return State_Info.format(self.Parameter)
    
    def __eq__(self, __value: object) -> bool:
        return True if isinstance(__value, State) and __value.Parameter == self.Parameter else False
        
    def __hash__(self):
        return hash(str(self.Parameter))
         
class TerminalState(State):
    def __init__(self, Parameter : int)->None:
        super().__init__(Parameter)
   
class StateSpace:
    def __init__(self):
        self.States = {}
        self.Agent: self.Agent()= None
    class Agent:
        def __init__(self, CurrentState: string_) -> None:
            self.CurrentState= CurrentState
            self.Trajectory: list
        def Policy(self)-> dict:
            pass
        def ChooseAction(self, StateProbabiliies: dict):
            Decision= random.choice(StateProbabiliies)
            
        def UpdateTrajectory(self, Action, NewState, Reward ):
            append(self.Trajectory, [Action, NewState, Reward])
        def __str__(self) -> str:
            return self.CurrentState
    def add_State(self, NewState: State):
        if isinstance(NewState, State)== False:
            raise Exception("Invalid State Type")
        elif NewState in self.States.values(): 
            raise Exception("State Duplicate")
        else:
            self.States["State{0}".format(len(self.States))]= NewState           
    def add_Connection(self, State1, State2):
        if State1 in self.States and State2 in self.States:
            self.States[State1].add_neighbor(self.States[State2])
            self.States[State2].add_neighbor(self.States[State1])
        else: 
            raise Exception("Invalid State Connection")
    def get_agent_state(self)-> string_:
        if self.Agent is None:
            print('No agent')
        elif isinstance(self.Agent, self.Agent()):
            return self.Agent.CurrentState
    def are_connected(self, State1: string_, State2: string_) -> bool:
        if self.States[State1] in self.States[State2].neighbors.values() or self.States[State2] in self.States[State1].neighbors.values():
            return True
        else : 
            return False
    def TransitionProbability(self, Action) -> dict:
        pass
    def StateTransition(self, Action: string_):
        self.Agent.CurrentState= self.Agent.CurrentState.neighbors[Action]
    def RewardFunction(self, Action, NextState) -> float:
        pass
    def __str__(self):
        StateSpace_str = ""
        StateName= list(self.States.keys())
        StateValues= list(self.States.values())
        for State in StateValues:
            neighborhood = ", ".join(StateName[StateValues.index(neighbor)] for neighbor in State.neighbors.values())
            StateSpace_str += f"{StateName[StateValues.index(State)]}: [{neighborhood}]\n"
        return StateSpace_str

def create_statespace(number_of_states: int32):
    NewStateSpace = StateSpace()
    while len(NewStateSpace.States) < number_of_states:
        NewState= State([random.randint(0,10), random.randint(0, 10)])
        if NewState not in NewStateSpace.States.values(): 
            NewStateSpace.add_State(NewState)
    return NewStateSpace

def connect_statespace(InputStateSpace: StateSpace, number_of_connections: int32):
    Connections= 0
    while Connections < number_of_connections:
        A= random.sample(list(InputStateSpace.States.keys()), k=2)
        if InputStateSpace.are_connected(A[0], A[1])== False:
            InputStateSpace.add_Connection(str(A[0]), str(A[1]))
            Connections+=1
    return InputStateSpace

StateSpace = create_statespace(10)
StateSpace= connect_statespace(StateSpace, 10)
# StateSpace.Agent= StateSpace.Agent(random.choice(list(StateSpace.States.keys())))
print(StateSpace)