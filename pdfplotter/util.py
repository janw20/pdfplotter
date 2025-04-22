from __future__ import annotations

import numpy as np
import sympy as sp
from elements import _pubchem_elements

from pdfplotter.flavors import *


def to_str(
    observable: sp.Basic,
    nucleus: str | None = None,
    multiply_x: bool = True,
    function_suffix: bool = True,
    x: float | None = None,
    Q: float | None = None,
    Q2: float | None = None,
    upright_flavors: bool = True,
    replace_valence: bool = True,
    R: bool = False,
) -> str:
    """Converts an observable to a LaTeX string, e.g. for using in plotting labels.

    Parameters
    ----------
    observable : sp.Basic
        The observable to get the string representation of.
    nucleus : str, optional
        Name of the nucleus of the PDFSet, you can get this by calling `pp.get_element_symbol(pdf_set.Z)`. Shows up as a superscript to the PDF label. By default None.
    multiply_x : bool, optional
        If the string representation should have a multiplication of `x` from the left. If the `x` would cancel out, i.e. in ratios of observables, it is automatically left out, so you should only set this to False if you don't want an `x` multiplication in non-ratios of observables. By default True.
    function_suffix : bool, optional
        If the string representation should have the suffix `(x, Q)`. Adds parentheses for ratios or sums, etc. By default True.

    Returns
    -------
    str
        String representation of `observable`.
    """

    uv_sym = sp.Symbol("uv")
    dv_sym = sp.Symbol("dv")

    x_coeff = sp.symbols("x")

    R_sym = sp.Symbol("R")

    flavors_ordered = [
        x_coeff,
        R_sym,
        g,
        u,
        ubar,
        uv_sym,
        d,
        dbar,
        dv_sym,
        s,
        sbar,
        c,
        cbar,
        b,
        bbar,
        t,
        tbar,
    ]

    if replace_valence:
        observable = observable.subs({u_v: uv_sym, d_v: dv_sym})

    if multiply_x and not R:
        observable = sp.factor(
            observable.subs(
                {sym: x_coeff * sym for sym in flavors_ordered if sym != x_coeff}
            )
        )

    # construct x0 x1 ... symbols (need leading zeros so sympy does the ordering correctly)
    symbols = map(
        lambda x: sp.Symbol(f"x{x:0{int(np.log(len(flavors_ordered)))}d}"),
        range(len(flavors_ordered)),
    )
    symbols_subs = dict(zip(flavors_ordered, symbols))
    # replace flavors with x0, x1, ... to ensure the correct ordering
    observable = observable.subs(symbols_subs)

    x_str = sp.latex(symbols_subs[x_coeff])

    if not R:
        result_str: str = (
            sp.latex(observable).replace("frac", "dfrac").replace(x_str, x_str + r" \,")
        )

        # add parentheses if they are needed
        needs_parentheses = (
            not result_str.startswith(x_str)
            and observable not in flavors_ordered
            and (nucleus is not None or function_suffix)
        )
        if needs_parentheses:
            if result_str.startswith(x_str):
                result_str = result_str.replace(x_str, x_str + r" \left(")
            else:
                result_str = r"\left(" + result_str
            result_str += r"\right)"
    else:

        # R with flavor subscript
        result_str: str = (
            "R_{"
            + (r"\mathrm{" if upright_flavors else "")
            + sp.latex(observable, fold_short_frac=True)
            + ("}" if upright_flavors else "")
            + "}"
        )

    # add nucleus superscript
    if nucleus is not None:
        space = "\\!" if result_str.endswith("\\right)") else ""
        result_str += rf"^{{{space}\mathrm{{{nucleus}}}}}"

    # surround the flavors with \mathrm
    if not R:
        if upright_flavors:
            if result_str.startswith(x_str):
                result_str = result_str.replace(x_str, x_str + r" \mathrm{") + "}"
            else:
                result_str = r"\mathrm{" + result_str + "}"

    # add function suffix
    if function_suffix:
        x_eq = f"x = {x}" if x is not None else "x"

        if Q is not None and Q2 is not None:
            raise ValueError("Only one of Q or Q2 must be given")

        Q_eq = (
            rf"Q = {Q}\,\mathrm{{GeV}}"
            if Q is not None
            else rf"Q^2 = {Q2}\,\mathrm{{GeV}}^2" if Q2 is not None else "Q"
        )
        xQ_suffix = rf"\left({x_eq}, {Q_eq}\right)"

        result_str += xQ_suffix

    # replace x0, x1, ... back with the flavors
    for f, f_sub in symbols_subs.items():
        result_str = result_str.replace(sp.latex(f_sub), sp.latex(f))

    if replace_valence:
        result_str = result_str.replace(sp.latex(uv_sym), r"u_\mathrm{v}").replace(
            sp.latex(dv_sym), r"d_\mathrm{v}"
        )

    return result_str


def get_element_symbol(Z: int) -> str:
    """Gets the symbol of an element from its atomic number. For `Z = 1`, the proton is assumed, so an empty string is returned.

    Parameters
    ----------
    Z : int
        Atomic number of the element.

    Returns
    -------
    str
        Symbol of the element.
    """
    return str(_pubchem_elements.loc[Z, "Symbol"]) if Z > 1 else ""
