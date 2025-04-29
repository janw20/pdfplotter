from __future__ import annotations

from itertools import zip_longest
from math import log
from typing import Sequence, Literal, Any

#from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.ticker as mticker
import matplotlib.cm as cm
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp

from pdfplotter.pdf_set import PDFSet
from pdfplotter.util import update_kwargs
from pdfplotter.util import log_tick_formatter
from pdfplotter import util


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

    def plot_A_dep_3d(
        self,
        ax: plt.Axes | npt.NDArray[plt.Axes],  # pyright: ignore[reportInvalidTypeForm]
        A: int | float | list[int | float],
        observables: (
            sp.Basic
            | npt.NDArray[sp.Basic]  # pyright: ignore[reportInvalidTypeForm]
            | list[sp.Basic]
        ),
        Q: float | None = None,
        Q2: float | None = None,
        colors: list = [],
        logA: bool = True,
        plot_uncertainty: bool = True,
        plot_ratio: bool = False,
        pdf_label: Literal["ylabel", "annotate"] = "annotate",
        A_label: Literal["legend", "ticks"] = "ticks",
        proj_type: Literal["ortho", "persp"] = "ortho",
        view_init: tuple[float, float] | list[tuple[float, float]] = (15, -75),
        kwargs_theory: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_uncertainty: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_uncertainty_edges: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_title: dict[str, Any] = {},
        kwargs_notation: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_ylabel: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_xlabel: dict[str, Any]  = {},
        kwargs_zlabel: dict[str, Any]  = {},
        kwargs_legend: dict[str, Any]  = {},
    ):

        my_sets = self.pdf_sets
        x = self.get(A=my_sets["A"][0]).x_values


        if Q is None and Q2 is None:
            raise ValueError("Please pass either `Q` or `Q2`")

        elif Q is not None and Q2 is not None:
            raise ValueError("Please pass either `Q` or `Q2`, not both")

        elif Q is not None:
            if Q not in self.get(A=my_sets["A"][0]).Q_values and Q not in np.sqrt(
                np.array(self.get(A=my_sets["A"][0]).Q2_values)
            ):
                raise ValueError(
                    f"Chosen Q value {Q} was not used for defining nuclear pdf set. \n Please choose Q that was used in initialization"
                )
        else:
            if (
                Q2 not in self.get(A=my_sets["A"][0]).Q2_values
                and Q2 not in np.array(self.get(A=my_sets["A"][0]).Q_values) ** 2
            ):
                raise ValueError(
                    f"Chosen Q2 value {Q2} was not used for defining nuclear pdf set. \n Please choose Q2 that was used in initialization"
                )

        if not isinstance(A, list):
            A = [A]

        if isinstance(observables, np.ndarray):
            observables = list(observables.flatten())

        if not isinstance(observables, list):
            observables = [observables]

        if not isinstance(ax, np.ndarray):
            ax = np.array([ax])

        if colors == []:
            cmap = cm.get_cmap("tab10", lut=len(A))
            colors = [cmap(i) for i in range(len(A))]

        for i, (obs_i, ax_i) in enumerate(zip(observables, ax.flat)):

            ax_i.set_proj_type(proj_type)
            ax_i.view_init(*view_init[i] if isinstance(view_init, list) else view_init)
            for j, (A_j, col_j) in enumerate(zip(A, colors)):
                z_lower, z_upper = self.get(A=A_j).get_uncertainties(
                    observable=obs_i, x=x, Q=Q, Q2=Q2
                )
                kwargs_default = {
                    "color": col_j,
                    "label": f"A={A_j}",
                    "linewidth": 1.5,
                }
                if not isinstance(kwargs_theory, list):
                    kwargs = update_kwargs(
                        kwargs_default,
                        kwargs_theory,
                    )
                else:
                    kwargs = update_kwargs(
                        kwargs_default,
                        kwargs_theory,
                        i=j,
                    )
                if logA:
                    ax_i.plot(
                        np.log10(x),
                        np.log10(len(x) * [A_j]),
                        self.get(A=A_j).get_central(x=x, Q=Q, Q2=Q2, observable=obs_i),
                        **kwargs,
                    )
                else:
                    ax_i.plot(
                        np.log10(x),
                        len(x) * [A_j],
                        self.get(A=A_j).get_central(x=x, Q=Q, Q2=Q2, observable=obs_i),
                        **kwargs,
                    )       
                if plot_uncertainty:
                    kwargs_uncertainty_default = {
                        "color": col_j,
                        "alpha": 0.3,
                    }
                    if not isinstance(kwargs_uncertainty, list):
                        kwargs = update_kwargs(
                            kwargs_uncertainty_default,
                            kwargs_uncertainty,
                        )
                    else:
                        kwargs = update_kwargs(
                            kwargs_uncertainty_default,
                            kwargs_uncertainty,
                            i=j,
                        )

                    vertices = []
                    z_lower = np.array(z_lower)
                    z_upper = np.array(z_upper)
                    if not logA:
                        for xi, ai, zl, zu in zip(
                            np.log10(x), np.ones(len(x)) * A_j, z_lower, z_upper
                        ):
                            vertices.append([xi, ai, zl])

                        for xi, ai, zl, zu in reversed(
                            list(zip(np.log10(x), np.ones(len(x)) * A_j, z_lower, z_upper))
                        ):
                            vertices.append([xi, ai, zu])
                    else:
                        for xi, ai, zl, zu in zip(
                            np.log10(x), np.ones(len(x)) * np.log10(A_j), z_lower, z_upper
                        ):
                            vertices.append([xi, ai, zl])

                        for xi, ai, zl, zu in reversed(
                            list(zip(np.log10(x), np.ones(len(x)) * np.log10(A_j), z_lower, z_upper))
                        ):
                            vertices.append([xi, ai, zu])                        
                    poly = Poly3DCollection([vertices], **kwargs)
                    ax_i.add_collection3d(poly)

                    kwargs_uncertainty_edges_default = {
                        "color": col_j,
                        "alpha": 1,
                    }
                    if not isinstance(kwargs_uncertainty_edges, list):
                        kwargs = update_kwargs(
                            kwargs_uncertainty_edges_default,
                            kwargs_uncertainty_edges,
                        )
                    else:
                        kwargs = update_kwargs(
                            kwargs_uncertainty_edges_default,
                            kwargs_uncertainty_edges,
                            i=j,
                        )
                    if not logA:
                        ax_i.plot(np.log10(x), len(x) * [A_j], z_upper, **kwargs)
                        ax_i.plot(np.log10(x), len(x) * [A_j], z_lower, **kwargs)
                    else:
                        ax_i.plot(np.log10(x), len(x) * [np.log10(A_j)], z_upper, **kwargs)
                        ax_i.plot(np.log10(x), len(x) * [np.log10(A_j)], z_lower, **kwargs)
            if pdf_label == "annotate":
                kwargs_notation_default = {
                    "fontsize": 12,
                    "xy": (0.97, 0.96),
                    "xycoords": "axes fraction",
                    "va": "top",
                    "ha": "right",
                    "bbox": dict(
                        facecolor=(1, 1, 1),
                        edgecolor=(0.8, 0.8, 0.8),
                        lw=0.9,
                        boxstyle="round, pad=0.2",
                    ),
                }
                if not isinstance(kwargs_notation, list):
                    kwargs_n = update_kwargs(
                        kwargs_notation_default,
                        kwargs_notation,
                    )
                else:
                    kwargs_n = update_kwargs(
                        kwargs_notation_default,
                        kwargs_notation,
                        i=i,
                    )
                ax_i.annotate(f"${util.to_str(obs_i, Q=Q,Q2=Q2)}$", **kwargs_n)

            if pdf_label == "ylabel":
                kwargs_ylabel_default = {
                    "fontsize": 14,
                    "zlabel": f"${util.to_str(obs_i,Q=Q,Q2=Q2)}$",
                    #"labelpad":-200
                }
                if not isinstance(kwargs_ylabel, list):
                    kwargs = update_kwargs(
                        kwargs_ylabel_default,
                        kwargs_ylabel,
                    )
                else:
                    kwargs = update_kwargs(
                        kwargs_ylabel_default,
                        kwargs_ylabel,
                        i=i,
                    )
                ax_i.set_zlabel(**kwargs)

            else:
                kwargs_notation_default = {
                    "fontsize": 12,
                    "xy": (0.47, 0.96),
                    "xycoords": "axes fraction",
                    "va": "top",
                    "ha": "right",
                    "bbox": dict(
                        facecolor=(1, 1, 1),
                        edgecolor=(0.8, 0.8, 0.8),
                        lw=0.9,
                        boxstyle="round, pad=0.2",
                    ),
                }
                kwargs_n = update_kwargs(kwargs_notation_default, kwargs_notation, i=i)

                ax_i.annotate(f"${util.to_str(obs_i, Q=Q,Q2=Q2)}$", **kwargs_n)
            ax_i.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter)) 
            if A_label == "ticks":
                if logA:
                    ax_i.set_yticks(np.log10(A),A)               
                else:
                    ax_i.set_yticks(A,A) 
                kwargs_zlabel_default = {
                    "fontsize": 14,
                    "ylabel": f"$A$",

                }
                kwargs = update_kwargs(
                    kwargs_zlabel_default,
                    kwargs_zlabel,
                )

                ax_i.set_ylabel(**kwargs)
            else:
                ax_i.set_yticks([])
                kwargs_legend_default = {
                    "fontsize": 12,
                    "bbox_to_anchor": (0.95, 0.95),
                    "frameon": False,
                }
                kwargs = update_kwargs(
                    kwargs_legend_default,
                    kwargs_legend,
                )
                ax_i.legend()
                kwargs_zlabel_default = {
                    "fontsize": 14,
                    "ylabel": f"$A$",
                    "labelpad":-10
                }
                kwargs = update_kwargs(
                    kwargs_zlabel_default,
                    kwargs_zlabel,
                )
                ax_i.set_ylabel(**kwargs)
            ax_i.xaxis.pane.fill=False   
            ax_i.yaxis.pane.fill=False   
            ax_i.zaxis.pane.fill=False   
            ax_i.xaxis.pane.set_edgecolor("white")
            ax_i.yaxis.pane.set_edgecolor("white")
            ax_i.zaxis.pane.set_edgecolor("white")

            ax_i.zaxis._axinfo["juggled"]=(1,2,0)

            kwargs_xlabel_default = {
                "fontsize": 14,
                "xlabel": f"$x$",

            }
            kwargs = update_kwargs(
                kwargs_xlabel_default,
                kwargs_xlabel,
            )
            ax_i.set_xlabel(**kwargs)