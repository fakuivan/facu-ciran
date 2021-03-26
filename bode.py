#!/usr/bin/env python3.8
from typing import Dict, Iterator, Optional, Tuple, NamedTuple, DefaultDict, Optional, Union
from collections import defaultdict
from sympy import Basic, Symbol, I
from sympy import conjugate, im, re, Piecewise, pi, log, And, Eq, Mul
from itertools import combinations, chain
from functools import reduce
from operator import mul

class Zero(NamedTuple):
    at: Basic
    order: int

"Maps the zero point to an oder"
Zeros = DefaultDict[Basic, int]

class Rational(NamedTuple):
    """
    Rational function represented using zeros, poles and a scale factor
    Poles are zeros with negative order
    """
    zeros: Zeros
    scale: Basic

    @classmethod
    def from_dict(cls, zeros: Dict[Basic, int], scale: Basic):
        return cls(defaultdict(int, zeros), scale)

    def as_rational(self, var: Symbol) -> Basic:
        return reduce(mul,
            ((var - point)**order for point, order in self.zeros.items()), self.scale)

    def normalized(self, var: Symbol) -> Basic:
        "an attempt at expressing variables in every zero scaled by it"
        factor = 1
        expr = 1
        for point, order in self.zeros.items():
            is_zero = Eq(point, 0)
            factor *= Piecewise((1           , is_zero),
                                (point**order, True))
            expr *= Piecewise((var**order           , is_zero), 
                             ((var/point - 1)**order, True))
        return factor * expr


def are_non_trivial_conjugates(a: Basic, b: Basic) -> bool:
    """
    non trivial conjugates are conjugate pairs
    where none of them is the conjugate of itself
    """
    return conjugate(a) == b and im(a) != 0


def split_conjugates(rational: Zeros, order: Optional[int] = None) -> \
        Tuple[Zeros, Zeros]:
    "moves conjugate pairs of the same order into a separate Zeros set"
    rat_list = list(rational.items())
    # Using combinations is the right choice here since
    # all conjugate operations are commutative
    allowed_order = lambda o: o == order if order is not None else True
    conj_pairs = [(i, j) for i, j in combinations(range(len(rat_list)), 2) if \
        rat_list[i][1] == rat_list[j][1] and allowed_order(rat_list[i][1]) and \
            are_non_trivial_conjugates(rat_list[i][0], rat_list[j][0])]

    conjugates = dict()
    for i, j in conj_pairs:
        point, order = rat_list[i]
        conjugates[point] = order

    non_conjugates = (rat_list[i] for i in range(len(rat_list)) \
        if i not in chain.from_iterable(conj_pairs))
    
    return defaultdict(int, conjugates), defaultdict(int, non_conjugates)


# These return their contribution to the "k" constant as the first element
# and a tuple as the last element with the linear asymptotic aproximations

def real_null_simple_zero(n: int, w: Symbol) -> \
        Tuple[Basic, Tuple[Basic, Basic]]:
    return 1, (w**n, (pi/2)*n)


def real_negative_simple_zero(point: Basic, w: Symbol) -> \
        Tuple[Basic, Tuple[Basic, Basic]]:
    break_frequency = -point
    module = Piecewise((1,                 w < break_frequency),
                       (w/break_frequency, w >= break_frequency))

    lower_frequencies = break_frequency / 10
    higher_frecuencies = break_frequency * 10
    phase_slope = (pi/4)
    phase = Piecewise((0   , w < lower_frequencies),
                      (log(w/lower_frequencies) * phase_slope, 
                             And(lower_frequencies <= w, w < higher_frecuencies)),
                      (pi/2, w >= higher_frecuencies))
    return break_frequency, (module, phase)

def real_negative_nth(point: Basic, order: int, w: Symbol) -> \
        Tuple[Basic, Tuple[Basic, Basic]]:
    lt0 = lambda expr, var: Piecewise((expr, var < 0))
    break_frequency = -point
    module = Piecewise((1,                 w < break_frequency),
                       ((w/break_frequency)**order, w >= break_frequency))

    lower_frequencies = break_frequency / 10
    higher_frecuencies = break_frequency * 10
    phase_slope = (pi/4)
    phase = Piecewise((0   , w < lower_frequencies),
                      (order*log(w/lower_frequencies, 10) * phase_slope, 
                             And(lower_frequencies <= w, w < higher_frecuencies)),
                      (order*(pi/2), w >= higher_frecuencies))
    return lt0(break_frequency, point), (lt0(module, point), lt0(phase, point))


def real_negative_simple_pole(point: Basic, w: Symbol) -> \
        Tuple[Basic, Tuple[Basic, Basic]]:
    break_frequency, (module, phase) = \
        real_negative_simple_zero(point, w)
    return 1/break_frequency, (1/module, -phase)


def complex_conjugate_simple_zero(point: Basic, w: Symbol) -> \
        Tuple[Basic, Basic, Tuple[Basic, Basic]]:
    """
    returns w0, gamma, asymptotic aproximation on the log plane for abs
              , same for arg
    """
    # We could implement some of these
    # https://lpsa.swarthmore.edu/Bode/underdamped/underdampedApprox.html

    natural_frequency = abs(point)
    damping = -re(point) / natural_frequency
    max_ = pi
    module = Piecewise((1                       , w < natural_frequency), 
                       ((w/natural_frequency)**2, w >= natural_frequency))
    phase = Piecewise((0   , w < natural_frequency),
                      (max_, w >= natural_frequency))

    return natural_frequency**2, damping, (module, phase)


def complex_conjugate_simple_pole(point: Basic, w: Symbol) -> \
        Tuple[Basic, Basic, Tuple[Basic, Basic]]:
    natural_frequency_sq, damping, (module, phase) = \
        complex_conjugate_simple_zero(point, w)
    return 1/natural_frequency_sq, damping, (1/module, -phase)