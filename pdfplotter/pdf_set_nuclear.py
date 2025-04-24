from __future__ import annotations

from itertools import cycle, zip_longest
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import sympy as sp
from matplotlib.lines import Line2D

from pdfplotter.elements import element_to_str
from pdfplotter.pdf_set import PDFSet
from pdfplotter.util import to_str, update_kwargs


class NuclearPDFSet(PDFSet):

    _pdf_sets: pd.DataFrame

    def __init__(
        self,
        x: npt.ArrayLike,
        Q: npt.ArrayLike | None = None,
        Q2: npt.ArrayLike | None = None,
        names: str | list[str] | None = None,
        name_format: str | None = None,
        name_regex: str | None = None,
        A: int | float | list[int | float] = 1,
        Z: int | float | list[int | float] = 1,
        construct_full_nuclear_pdfs: bool = False,
        confidence_level: float = 90,
    ) -> None:
        """Constructs a NuclearPDFSet object, wrapping multiple PDFSet objects for different values of A.

        Parameters
        ----------
        x : numpy.ArrayLike
            The momentum fraction values at which the PDFs are evaluated.
        Q : numpy.ArrayLike, optional
            The scale values at which the PDFs are evaluated. Either `Q` or `Q2` must be given.
        Q2 : numpy.ArrayLike, optional
            The Q² (squared scale) values at which the PDFs are evaluated. Either `Q` or `Q2` must be given.
        names : str | list[str], optional
            Name(s) of the LHAPDF sets to load. One of `names` and `name_format` must be given.
        name_format : str, optional
            Format string for the names of the LHAPDF sets to load, e.g. "nCTEQ15HQ_{A}_{Z}" (don't add an `f` prefix). The values for A and/or Z must be passed as `A` and `Z`, respectively. One of `names` and `name_format` must be given.
        name_regex : str, optional
            Regular expression to match the names of the LHAPDF sets to load. Not implemented yet.
        A : int | float | list[int | float], optional
            Mass number of the nucleus, by default 1
        Z : int, optional
            Charge number of the nucleus, by default 1
        construct_full_nuclear_pdfs : bool, optional
            True if full nuclear PDFs should be constructed using `xf^A = Z/A * xf^(p/A) + (A - Z)/A * xf^(n/A). Only use this if the LHAPDF set you are loading does not already load full nuclear PDFs. By default False.
        confidence_level : float, optional
            The confidence level in percent at which the uncertainties are calculated. By default 90.
        """

        if names is None:
            if name_format is None:
                raise ValueError("Please pass either `names` or `name_format`")

            names = []

        elif isinstance(names, str):
            names = [names]

        if name_regex is not None:
            raise NotImplementedError("`name_regex` is not implemented yet")

        if not isinstance(A, list):
            A = [A]

        if not isinstance(Z, list):
            Z = [Z]

        if len(A) != len(Z):
            raise ValueError("A and Z must have the same length")

        if len(A) < len(names):
            raise ValueError("names must have the same length as A and Z")

        pdf_sets_dict = []

        for A_i, Z_i, name_i in zip_longest(A, Z, names):
            assert A_i is not None
            assert Z_i is not None

            if name_i is None:
                if name_format is None:
                    raise ValueError(
                        "Please pass `name_format` if `names` has less elements than `A` and `Z`"
                    )
                name_i = name_format.format(A=A_i, Z=Z_i)

            pdf_sets_dict.append(
                {
                    "A": A_i,
                    "Z": Z_i,
                    "pdf_set": PDFSet(
                        name_i,
                        x=x,
                        Q=Q,
                        Q2=Q2,
                        A=A_i,  # pyright: ignore[reportArgumentType]
                        Z=Z_i,  # pyright: ignore[reportArgumentType]
                        construct_full_nuclear_pdfs=construct_full_nuclear_pdfs,
                        confidence_level=confidence_level,
                    ),
                }
            )

        self._pdf_sets = pd.DataFrame(
            pdf_sets_dict,
        )

    @property
    def pdf_sets(self) -> pd.DataFrame:
        """The `PDFSet` objects"""
        return self._pdf_sets

    def get(
        self, *, A: int | float | None = None, Z: int | float | None = None
    ) -> PDFSet:
        """Get the PDFSet for a given A or Z.

        Parameters
        ----------
        A : int | float | None, optional
            Mass number of the nucleus, by default None. One of `A` and `Z` must be given.
        Z : int | float | None, optional
            Charge number of the nucleus, by default None. One of `A` and `Z` must be given.

        Returns
        -------
        PDFSet
            The PDFSet for the given A or Z.
        """
        if A is None and Z is None:
            raise ValueError("Please pass either `A` or `Z`")
        elif A is not None and Z is not None:
            raise ValueError("Please pass either `A` or `Z`, not both")
        elif A is not None:
            pdf_set = self.pdf_sets[self.pdf_sets["A"] == A]
            if pdf_set.shape[0] == 0:
                raise ValueError(f"No PDFSet found for A = {A}")
            elif pdf_set.shape[0] > 1:
                raise ValueError(f"Multiple PDFSets found for A = {A}")
            else:
                return pdf_set.iloc[0]["pdf_set"]
        else:
            pdf_set = self.pdf_sets[self.pdf_sets["Z"] == Z]
            if pdf_set.shape[0] == 0:
                raise ValueError(f"No PDFSet found for Z = {Z}")
            elif pdf_set.shape[0] > 1:
                raise ValueError(f"Multiple PDFSets found for Z = {Z}")
            else:
                return pdf_set.iloc[0]["pdf_set"]

    def plot_A_dep(
        self,
        ax: plt.Axes | npt.NDArray[plt.Axes],  # pyright: ignore[reportInvalidTypeForm]
        observable: sp.Basic | list[sp.Basic],
        A: list[int | float] | None = None,
        x: npt.ArrayLike | None = None,
        Q: npt.ArrayLike | None = None,
        Q2: npt.ArrayLike | None = None,
        pdf_label: Literal["ylabel", "annotate"] = "ylabel",
        legend: bool = True,
        kwargs_central: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_uncertainty: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_uncertainty_edges: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_annotation: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_xlabel: dict[str, Any] = {},
        kwargs_ylabel: dict[str, Any] = {},
        kwargs_legend: dict[str, Any] = {},
    ) -> None:
        if not isinstance(ax, np.ndarray):
            ax = np.array([ax])

        if not isinstance(observable, list):
            observable = [observable]

        if A is None:
            A = list(self.pdf_sets["A"])
        elif not isinstance(A, list):
            A = [A]

        # plot each observable on its own subplot
        for obs_i, ax_i in zip(observable, ax.flat):
            ax_i: plt.Axes

            # get the colors from the default color cycle so individual colors for each A are overridable
            colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

            # we save the lines to add them to the legend in the end
            lines = []

            for i, A_i in enumerate(A):

                pdf_set = self.get(A=A_i)
                x_i = x if x is not None else pdf_set.x_values
                Q_i = Q if Q is not None else pdf_set.Q_values

                Q_i_label = None if isinstance(Q_i, np.ndarray) else cast(float, Q_i)

                color = next(colors)

                # plot the central PDF
                kwargs_default = {
                    "zorder": 15,
                    "color": color,
                }
                kwargs = update_kwargs(
                    kwargs_default,
                    kwargs_central,
                    i,
                )
                l = ax_i.plot(
                    x_i,
                    pdf_set.get_central(obs_i, Q=Q_i),
                    label=f"$A = {A}$ ({element_to_str(A_i)})",
                    **kwargs,
                )
                lines.append(l[0])

                # plot the uncertainty band
                kwargs_default = {
                    "zorder": 15,
                    "alpha": 0.3,
                    "facecolor": color,
                }
                kwargs = update_kwargs(
                    kwargs_default,
                    kwargs_uncertainty,
                    i,
                )
                ax_i.fill_between(
                    x_i,
                    *pdf_set.get_uncertainties(observable=obs_i, Q=Q_i),
                    **kwargs,
                )  # pyright: ignore[reportArgumentType]

                # plot the edges of the uncertainty band
                kwargs_default = {
                    "zorder": 15,
                    "color": color,
                    "lw": 0.5,
                }
                kwargs = update_kwargs(
                    kwargs_default,
                    kwargs_uncertainty_edges,
                    i,
                )
                ax_i.plot(
                    x_i,
                    pdf_set.get_uncertainty("+", obs_i, Q=Q_i),
                    **kwargs,
                )
                ax_i.plot(
                    x_i,
                    pdf_set.get_uncertainty("-", obs_i, Q=Q_i),
                    **kwargs,
                )

                ax_i.set_xscale("log")
                ax_i.set_xlim(1e-5, 1)

                if i == 0:
                    # add the pdf_label as annotation ...
                    if pdf_label == "annotate":
                        kwargs_default = {
                            "xy": (0.97, 0.95),
                            "xycoords": "axes fraction",
                            "va": "top",
                            "ha": "right",
                            "fontsize": 11,
                            "bbox": dict(
                                facecolor=(1, 1, 1),
                                edgecolor=(0.8, 0.8, 0.8),
                                lw=0.9,
                                boxstyle="round,pad=0.2",
                            ),
                        }
                        kwargs = update_kwargs(
                            kwargs_default,
                            kwargs_annotation,
                            i,
                        )
                        ax_i.annotate(
                            f"${to_str(obs_i, Q=Q_i_label)}$",
                            **kwargs,
                        )
                    # ... or as ylabel
                    elif pdf_label == "ylabel":
                        ax_i.set_ylabel(
                            f"${to_str(obs_i, Q=Q_i_label)}$", **kwargs_ylabel
                        )

                    ax_i.set_xlabel("$x$", **kwargs_xlabel)

            # add the legend
            if legend:
                kwargs_default = {
                    "loc": "upper left",
                    "bbox_to_anchor": (0.63, 1.02),
                }
                kwargs = update_kwargs(
                    kwargs_default,
                    kwargs_legend,
                )
                ax_i.figure.legend(
                    lines,
                    [f"$A = {A_i}$ ({element_to_str(A_i)})" for A_i in A],
                    **kwargs,
                )

    def plot_A_dep_pseudo_3d(
        self,
        ax: plt.Axes | npt.NDArray[plt.Axes],  # pyright: ignore[reportInvalidTypeForm]
        observable: sp.Basic | list[sp.Basic],
        A: list[int | float] | None = None,
        x: npt.ArrayLike | None = None,
        Q: npt.ArrayLike | None = None,
        Q2: npt.ArrayLike | None = None,
        offset: tuple[float, float] = (0.15, 0.1),
        pdf_label: Literal["ylabel", "annotate"] = "ylabel",
        legend: bool = True,
        kwargs_central: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_uncertainty: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_uncertainty_edges: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_spines: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_annotation: dict[str, Any] = {},
        kwargs_xlabel: dict[str, Any] = {},
        kwargs_ylabel: dict[str, Any] = {},
        kwargs_legend: dict[str, Any] = {},
    ) -> None:
        if not isinstance(ax, np.ndarray):
            ax = np.array([ax])

        if not isinstance(observable, list):
            observable = [observable]

        if A is None:
            A = list(self.pdf_sets["A"])
        elif not isinstance(A, list):
            A = [A]

        # plot each observable on its own subplot
        for obs_i, ax_i in zip(observable, ax.flat):
            ax_i: plt.Axes

            # since we draw the spines manually, the y limits need to be the same for the insets. so we determine the upper limit from the maximum beforehand
            ylim = (
                0,
                np.max(
                    [
                        self.get(A=A_i).get_uncertainty("+", obs_i, x=x, Q=Q, Q2=Q2)
                        for A_i in A
                    ]
                ),
            )

            # get the colors from the default color cycle so individual colors for each A are overridable
            colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

            # we save the lines to add them to the legend in the end
            lines = []

            for i, A_i in enumerate(A):

                # plot on insets in every but the first iteration
                if i != 0:
                    ax_old = ax_i
                    ax_i = ax_old.inset_axes([*offset, 1, 1], zorder=10)
                else:
                    ax_old = None

                pdf_set = self.get(A=A_i)
                x_i = x if x is not None else pdf_set.x_values
                Q_i = Q if Q is not None else pdf_set.Q_values

                Q_i_label = None if isinstance(Q_i, np.ndarray) else cast(float, Q_i)

                color = next(colors)

                # plot the central PDF
                kwargs_default = {
                    "zorder": 15,
                    "color": color,
                }
                kwargs = update_kwargs(
                    kwargs_default,
                    kwargs_central,
                    i,
                )
                l = ax_i.plot(
                    x_i,
                    pdf_set.get_central(obs_i, Q=Q_i),
                    label=f"$A = {A}$ ({element_to_str(A_i)})",
                    **kwargs,
                )
                lines.append(l[0])

                # plot the uncertainty band
                kwargs_default = {
                    "zorder": 15,
                    "alpha": 0.3,
                    "facecolor": color,
                }
                kwargs = update_kwargs(
                    kwargs_default,
                    kwargs_uncertainty,
                    i,
                )
                ax_i.fill_between(
                    x_i,
                    *pdf_set.get_uncertainties(observable=obs_i, Q=Q_i),
                    **kwargs,
                )  # pyright: ignore[reportArgumentType]

                # plot the edges of the uncertainty band
                kwargs_default = {
                    "zorder": 15,
                    "color": color,
                    "lw": 0.5,
                }
                kwargs = update_kwargs(
                    kwargs_default,
                    kwargs_uncertainty_edges,
                    i,
                )
                ax_i.plot(
                    x_i,
                    pdf_set.get_uncertainty("+", obs_i, Q=Q_i),
                    **kwargs,
                )
                ax_i.plot(
                    x_i,
                    pdf_set.get_uncertainty("-", obs_i, Q=Q_i),
                    **kwargs,
                )

                # hide top and right spines because they are in the way of the inset
                ax_i.spines["top"].set(visible=False)
                ax_i.spines["right"].set(visible=False)

                ax_i.set_xscale("log")
                ax_i.set_xlim(1e-5, 1)
                ax_i.set_ylim(*ylim)  # pyright: ignore[reportArgumentType]

                if i != 0:
                    assert ax_old is not None

                    kwargs_default = {
                        "zorder": -1,
                        "color": "gray",
                        "lw": 0.5,
                    }
                    kwargs = update_kwargs(
                        kwargs_default,
                        kwargs_spines,
                        i,
                    )

                    # draw the spines through the y ticks
                    for t in ax_old.get_yticks():
                        # transform the y tick to the y axis coordinates so that `offset` is in axis coordinates
                        # fmt: off
                        t_l = (ax_old.transScale + ax_old.transLimits).transform(  # pyright: ignore[reportAttributeAccessIssue]
                            [0, t]
                        )[1]
                        # fmt: on

                        # if the tick is not on the axis just leave it out
                        if t_l < 0 or t_l > 1:
                            continue

                        # add the ticks as lines
                        ax_old.add_artist(
                            Line2D(
                                [0, offset[0]],
                                [t_l, offset[1] + t_l],
                                transform=ax_old.transAxes,
                                figure=ax_old.figure,
                                clip_on=False,
                                **kwargs,
                            )
                        )

                    for t in ax_old.get_xticks():
                        # transform the x tick to the x axis coordinates so that `offset` is in axis coordinates
                        # fmt: off
                        t_l = (ax_old.transScale + ax_old.transLimits).transform(  # pyright: ignore[reportAttributeAccessIssue]
                            [t, 0]
                        )[0]
                        # fmt: on

                        # if the tick is not on the axis just leave it out
                        if t_l < 0 or t_l > 1:
                            continue

                        # add the ticks as lines
                        ax_old.add_artist(
                            Line2D(
                                [t_l, offset[0] + t_l],
                                [0, offset[1]],
                                transform=ax_old.transAxes,
                                figure=ax_old.figure,
                                **kwargs,
                            )
                        )

                    ax_i.set_xticklabels([])
                    ax_i.set_yticklabels([])

                    # hide the axis patch so that the insets are visible
                    # fmt: off
                    ax_i.patch.set_alpha(0)  # pyright: ignore[reportAttributeAccessIssue]
                    # fmt: on

                    # add the ylabel as annotation
                    if pdf_label == "annotate" and i == len(A) - 1:
                        kwargs_default = {
                            "xy": (0.05, 0.95),
                            "xycoords": "axes fraction",
                            "va": "top",
                            "ha": "left",
                            "fontsize": 11,
                            "bbox": dict(
                                facecolor=(1, 1, 1),
                                edgecolor=(0.8, 0.8, 0.8),
                                lw=0.9,
                                boxstyle="round,pad=0.2",
                            ),
                        }
                        kwargs = update_kwargs(
                            kwargs_default,
                            kwargs_annotation,
                        )
                        ax_i.annotate(
                            f"${to_str(obs_i, Q=Q_i_label)}$",
                            **kwargs,
                        )

                else:
                    ax_i.set_xlabel("$x$", **kwargs_xlabel)
                    ax_i.set_ylim(0)

                    # add the ylabel
                    if pdf_label == "ylabel":
                        ax_i.set_ylabel(
                            f"${to_str(obs_i, Q=Q_i_label)}$", **kwargs_ylabel
                        )

            # add the legend
            if legend:
                kwargs_default = {
                    "loc": "upper left",
                    "bbox_to_anchor": (0.63, 1.02),
                }
                kwargs = update_kwargs(
                    kwargs_default,
                    kwargs_legend,
                )
                ax_i.figure.legend(
                    lines,
                    [f"$A = {A_i}$ ({element_to_str(A_i)})" for A_i in A],
                    **kwargs,
                )
