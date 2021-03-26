from utils import counter_wrap
from more_itertools import consume, pairwise, spy#, grouper, unzip
from typing import Iterable
from sympy import Basic

def routh_row(i_n_minus_2: Iterable[Basic],
              i_n_minus_1: Iterable[Basic]) -> \
        Iterable[Basic]:
    "Computes the next row for a Routh matrix"
    pp_iter, pp_counter = counter_wrap(i_n_minus_2)
    p_iter, p_counter = counter_wrap(i_n_minus_1)
    a02, pp_iter = spy(pp_iter, 2)
    a1, p_iter = spy(p_iter, 1)

    for (a0, a2), (a1, a3) in zip(
            pairwise(pp_iter), pairwise(p_iter)):
        yield (a1*a2 - a0*a3)/a1
    consume(map(consume, (pp_iter, p_iter)))
    if pp_counter() == 2 and p_counter() == 1:
        yield a02[1]
        return
    if not 0 <= pp_counter() - p_counter() <= 1 \
       or p_counter() < 1:
        raise ValueError(
            "pp row should be at most one item "
            "larger than p row and at least equal in size")

#def routh_matrix(coeffs: Iterable[Basic]) ->
#        Iterable[List[Basic]]:
#    coeffs, coeffs_n = counter_wrap(coeffs)
#    i0, i1 = map(list, unzip(grouper(coeffs, 2, 0)))
#    i2: List[Basic]
#    for _ in range(coeffs_n() - 2):

#def routh_recursive(coeffs: )