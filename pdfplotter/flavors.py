from __future__ import annotations

from typing_extensions import cast

import sympy as sp

flavors: tuple[sp.Symbol, ...] = sp.symbols("d d̅ u u̅ s s̅ c c̅ b b̅ t t̅ g")
d, dbar, u, ubar, s, sbar, c, cbar, b, bbar, t, tbar, g = flavors

u_v: sp.Basic = u - ubar  # pyright: ignore[reportOperatorIssue]
d_v: sp.Basic = d - dbar  # pyright: ignore[reportOperatorIssue]

flavor_from_pid: dict[int, sp.Basic] = {
    1: d,
    -1: dbar,
    2: u,
    -2: ubar,
    3: s,
    -3: sbar,
    4: c,
    -4: cbar,
    5: b,
    -5: bbar,
    6: t,
    -6: tbar,
    21: g,
}
pid_from_flavor: dict[sp.Basic, int] = dict((v, k) for (k, v) in flavor_from_pid.items())
anti_flavor: dict[sp.Basic, sp.Basic] = {
    d: dbar,
    dbar: d,
    u: ubar,
    ubar: u,
    s: sbar,
    sbar: s,
    c: cbar,
    cbar: c,
    b: bbar,
    bbar: b,
    t: tbar,
    tbar: t,
    g: g,
}
isospin_transform: dict[sp.Basic, sp.Basic] = {
    d: u,
    dbar: ubar,
    u: d,
    ubar: dbar,
    s: c,
    sbar: cbar,
    c: s,
    cbar: sbar,
    b: t,
    bbar: tbar,
    t: b,
    tbar: bbar,
}
flavors_nucleus: set[sp.Basic] = {d, dbar, u, ubar}


def add_flavor(flavor: sp.Basic, id: int) -> None:
    """Adds a custom flavor.

    Parameters
    ----------
    flavor : sp.Basic
        Sympy symbol of the new flavor.
    id : int
        ID of the new flavor.
    """
    global flavor_from_pid, pid_from_flavor
    flavor_from_pid[id] = flavor
    pid_from_flavor[flavor] = id
