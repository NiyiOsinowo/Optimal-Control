from dataclasses import dataclass
from typing import Optional, Mapping, Sequence, Tuple
from collections import Counter
import numpy as np
from numpy.random import binomial
import itertools
from operator import itemgetter
from matplotlib.pyplot import *
@dataclass
class Process1:
    @dataclass
    class State:
        price: int

    level_param: int  # level to which price mean-reverts
    alpha1: float = 0.25  # strength of mean-reversion (non-negative value)

    def up_prob(self, state: State) -> float:
        return  1. / (1 + np.exp(-self.alpha1 * (self.level_param - state.price)))

    def next_state(self, state: State) -> State:
        up_move: int = binomial(1, self.up_prob(state), 1)[0]
        return Process1.State(price=state.price + up_move * 2 - 1)

def simulation(process, start_state):
    state = start_state
    while True:
        yield state
        state = process.next_state(state)

def process1_price_traces(
        start_price: int,
        level_param: int,
        alpha1: float,
        time_steps: int,
        num_traces: int
        ) -> np.ndarray:
     process = Process1(level_param=level_param, alpha1=alpha1)
     start_state = Process1.State(price=start_price)
     return np.vstack([
         np.fromiter((s.price for s in itertools.islice(
         simulation(process, start_state),
         time_steps + 1
         )), float) for _ in range(num_traces)])
btr= process1_price_traces(0, 0, 0.5, 200, 3)
s = np.random.binomial(10, 0.1, 1)[0]
print(s)

