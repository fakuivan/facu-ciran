#!/usr/bin/env python3.8
from functools import reduce
from operator import mul
from typing import Generic, Iterator, Tuple, Dict, Any, Optional, Type, TypeVar, Callable, Iterable, Union, cast
from more_itertools.recipes import unique_everseen
from sympy import plot as splot, Basic, log, Symbol, Piecewise, And, solve, \
                  Eq, together, numer, denom, roots, LC as leading_coeff, Expr, \
                  re, im, N
from sympy.plotting.plot import Plot
from pprint import pprint
from matplotlib import pyplot
from itertools import count, dropwhile, takewhile, tee, filterfalse, zip_longest
from more_itertools import pairwise, partition, side_effect, consume, peekable, unzip
from contextlib import contextmanager
import random
import colorsys

Curve = Tuple[Any, Dict[str, Any]]
Range = Tuple[Any, None]
CurveOrRange = Union[Curve, Range]

def curves_iter(curves: Iterable[CurveOrRange]
) -> Iterator[Curve]:
    return cast(Iterator[Curve], 
        filterfalse(lambda expr: expr[1] is None, curves))

def splot_multiple(
    *exprs: CurveOrRange,
    plotf: splot, random_colors=False,
    **options):
    "plots multiple curves with extra kwargs for each one"
    show = options.pop("show", True)
    curves, curves_args = unzip(curves_iter(exprs))
    #pprint(curves)
    plot = plotf(*curves, show=False, **options)
    for i, curve_args in enumerate(curves_args):
        if random_colors:
            plot[i].line_color = [random_bright_rgb_color()]
        for key, value in curve_args.items():
            setattr(plot[i], key, value)
    plot.show() if show else None
    return plot

# Stolen from https://stackoverflow.com/a/43437435/5538719
def random_bright_rgb_color():
    "Picks a random \"bright\" RGB color"
    h = random.random()
    s = 0.5 + random.random()/2.0
    l = 0.4 + random.random()/5.0
    return colorsys.hls_to_rgb(h, l, s)

# Stolen from https://stackoverflow.com/a/60325901/5538719
def move_sympyplot_to_axes(p: Plot, ax) -> None:
    backend = p.backend(p)
    backend.ax = ax
    # Fix for > sympy v1.5
    backend._process_series(backend.parent._series, ax, backend.parent)
    backend.ax.spines['right'].set_color('none')
    backend.ax.spines['bottom'].set_position('zero')
    backend.ax.spines['top'].set_color('none')
    pyplot.close(backend.fig)

def lambdify_raw(expr: Expr, symbols: Tuple[Symbol]):
    def lambdified(*values: float) -> float:
        return N(expr, subs=dict(zip(symbols, values)))
    return lambdified

def log10(x: Basic, **kwargs):
    return log(x, 10)

def dB(x: Basic):
    return 20*log10(x)

def xlog(y: Basic, x: Symbol, base: Basic = 10):
    "For log plots on the x axis"
    return y.subs(x, 10**x)

def f2nd(function: Basic) -> Tuple[Basic, Basic]:
    function = together(function)
    return numer(function), denom(function)

def f2zpk(function: Basic, var: Symbol) -> \
        Tuple[Dict[Basic, int], Dict[Basic, int], Basic]:
    numer, denom = f2nd(function)
    numer_lc, denom_lc = map(
        lambda expr: leading_coeff(expr, var),
        (numer, denom))
    # It's not really necesary to devide by the leading
    # coefficient, but what do I know
    return roots(numer/numer_lc, var), \
           roots(denom/denom_lc, var), numer_lc/denom_lc

def zpk2f(poles: Dict[Basic, int],
          zeros: Dict[Basic, int],
          scale_factor: Basic, var: Symbol):
    return reduce(mul, [(var - z)**e for z, e in zeros.items()], 1)/ \
           reduce(mul, [(var - p)**e for p, e in poles.items()], 1) * \
               scale_factor

def reim(expr: Expr):
    return re(expr), im(expr)

@contextmanager
def change_figure_size(horizontal: Optional[float],
                       vertical: Optional[float],
                       pyplot: pyplot):
    "Changes the figure size while in the scope of the ``with`` statement"
    prev_fig_size: Tuple[float, float] = pyplot.rcParams['figure.figsize']
    new_fig_size = [size if size is not None else prev_size \
        for size, prev_size in zip((horizontal, vertical), prev_fig_size)]
    pyplot.rcParams['figure.figsize'] = tuple(new_fig_size)
    try:
        yield
    finally:
        pyplot.rcParams['figure.figsize'] = prev_fig_size

def spike(var: Symbol,
          left: Basic,
          mid: Basic,
          right: Basic,
          value: Basic):
    fw_slope = -value/right
    bw_slope = value/left
    var=var-mid
    return \
        Piecewise(((var+left)*bw_slope, And(var > -left, var < 0)),
                  ((var-right)*fw_slope, And(var >= 0, var < right)),
                  (0, True))

# Linear interpolator
def spike_train(var: Symbol, points: Dict[Basic, Basic]):
    points_sorted = sorted(points.items(), key=lambda e: e[0])
    spikes = 0
    for i in range(1, len(points_sorted)-1):
        mid = points_sorted[i][0]
        left = mid - points_sorted[i-1][0]
        right = points_sorted[i+1][0] - mid
        value = points_sorted[i][1]
        spikes += spike(var, left, mid, right, value)
    
    left_start, left_value = points_sorted[0]
    left_step = points_sorted[1][0] - left_start
    spikes += Piecewise((left_value, var < left_start),
                        ((var-left_start-left_step)*
                            (-left_value/left_step), 
                                var < left_start + left_step),
                        (0, True))

    right_start, right_value = points_sorted[-1]
    right_step = right_start - points_sorted[-2][0]
    spikes += Piecewise((right_value, right_start < var),
                        ((var-right_start+right_step)*
                            (right_value/right_step),
                                right_start - right_step < var),
                        (0, True))
    return spikes

T = TypeVar("T")
def counter_wrap(iterable: Iterable[T]) -> \
        Tuple[Iterator[T], Callable[[], int]]:
    """
    Returns a new iterator based on ``iterable``
    and a getter that when called returns the number of times
    the returned iterator was called up until that time
    """
    counter = peekable(count())
    return (
        side_effect(lambda e: next(counter), iterable),
        lambda: counter.peek()
    )
