from __future__ import annotations

from itertools import zip_longest
from typing import Sequence, Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import math
import seaborn as sns
from pdfplotter import util
from pdfplotter.util import update_kwargs

from pdfplotter.pdf_set import PDFSet
from pdfplotter import flavors


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

    def plot_Af_plane(
        self,
        ax: plt.Axes | npt.NDArray[plt.Axes],  # pyright: ignore[reportInvalidTypeForm]
        x_vals: float | list[float],
        observables: (
            sp.Basic
            | npt.NDArray[sp.Basic]  # pyright: ignore[reportInvalidTypeForm]
            | list[sp.Basic] 
        ),
        Q: float = 2,
        colors: list = [],
        logx: bool = True,
        title: str | list[str] | None = None,
        plot_unc: bool = False,
        plot_ratio: bool = False,
        pdf_label: Literal["ylabel", "annotate"] = "annotate",
        kwargs_theory: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_legend: dict[str, Any] = {},
        kwargs_xlabel: dict[str, Any] = {},
        kwargs_ylabel: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_title: dict[str, Any] = {},
        kwargs_notation: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_uncertainty: dict[str, Any] | list[dict[str, Any] | None] = {},
    ) -> None:
        """Plot nuclear PDFs in the A-f plane for different values of x.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes | numpy.ndarray[matplotlib.axes.Axes]
            The axes to plot on.
        x_vals : float | list[float]
            The x values to plot for.
        observables : sympy.Basic | numpy.ndarray[sympy.Basic] | list[sympy.Basic]
            The observables to plot.
        Q : float, optional
            The scale at which to plot the PDFs, by default 2.
        colors : list, optional
            The colors to use for the different x values, by default [] and seaborn color palette viridis is used if == [].
        logx : bool, optional   
            If True, use a logarithmic scale for the x axis, by default True.
        title : str | list[str] | None, optional
            The title of the plot, by default None. If a list is passed, the titles are set for each subplot. If a single string is passed, it is set for the first subplot.
        plot_unc : bool, optional
            If True, plot the uncertainties, by default False.
        plot_ratio : bool, optional
            If True, plot the ratio of the PDFs to the Proon PDF, by default False.
        pdf_label : str, optional
            The label for the PDF, by default "annotate". If "ylabel", the label is set as the y-axis label. If "annotate", the label is set as an annotation in the top right corner of the plot.
        kwargs_theory : dict[str, Any] | list[dict[str, Any] | None], optional 
            The keyword arguments to pass to the plot function for the central PDF, by default {}.
        kwargs_legend : dict[str, Any], optional
            The keyword arguments to pass to the legend function, by default {}.
        kwargs_xlabel : dict[str, Any], optional
            The keyword arguments to pass to the xlabel function, by default {}.
        kwargs_ylabel : dict[str, Any] | list[dict[str, Any] | None], optional
            The keyword arguments to pass to the ylabel function, by default {}.
        kwargs_title : dict[str, Any], optional
            The keyword arguments to pass to the title function, by default {}.
        kwargs_notation : dict[str, Any] | list[dict[str, Any] | None], optional
            The keyword arguments to pass to the annotation function, by default {}.
        kwargs_uncertainty : dict[str, Any] | list[dict[str, Any] | None], optional
            The keyword arguments to pass to the fill_between function for the uncertainties, by default {}.
      
        """

        my_sets = self.pdf_sets
        my_data = {}

        if not isinstance(x_vals, list):
            x_vals = [x_vals]

        for x in x_vals:
            if x not in self.get(A=my_sets["A"][0]).x_values:
                raise ValueError(
                    f"Chosen x value {x} was not used for defining nuclear pdf set. \n Pleas choose x that was used in initialising"
                )

        if isinstance(observables, np.ndarray):
            observables = list(observables.flatten())

        if not isinstance(observables, list):
            observables = [observables]

        if colors == []:
            colors = sns.color_palette("viridis", n_colors=len(x_vals))

        if len(colors) != len(x_vals):
            raise ValueError("No. of colors must match no. of x-values")

        for obs in observables:
            data_obs = {}
            list_x = []
            list_central = []
            list_unc1 = []
            list_unc2 = []
            list_A = []
            for A in my_sets["A"]:
                for x in self.get(A=A).x_values:
                    list_x.append(x)
                    list_central.append(
                        self.get(A=A).get_central(x=x, Q=Q, observable=obs)
                    )
                    unc1 = self.get(A=A).get_uncertainties(x=x, Q=Q, observable=obs)[0]
                    unc2 = self.get(A=A).get_uncertainties(x=x, Q=Q, observable=obs)[1]
                    if math.isnan(unc1):
                        list_unc1.append(
                            self.get(A=A).get_central(x=x, Q=Q, observable=obs)
                        )
                    else:
                        list_unc1.append(unc1)
                    if math.isnan(unc2):
                        list_unc2.append(
                            self.get(A=A).get_central(x=x, Q=Q, observable=obs)
                        )
                    else:
                        list_unc2.append(unc2)
                i = 0
                while i < len(self.get(A=A).x_values):
                    list_A.append(A)
                    i += 1

            data_obs["A"] = list_A
            data_obs["x"] = list_x
            data_obs["central"] = list_central
            data_obs["unc1"] = list_unc1
            data_obs["unc2"] = list_unc2

            dataframe_obs = pd.DataFrame(data_obs)
            my_data[obs] = dataframe_obs

        # fig, ax = plt.subplots(1, len(observables), figsize=(9 * len(observables), 5))

        if not isinstance(ax, np.ndarray):
            ax = np.array([ax])

        for m, (obs_m, ax_m) in enumerate(zip(observables, ax.flat)):
            ax_m: plt.Axes

            if not plot_ratio:
                for j, (x, col) in enumerate(zip(x_vals, colors)):
                    kwargs_default = {
                        "color": col,
                        "label": f"x={x}",
                        "linewidth": 1.5,
                    }
                    kwargs = update_kwargs(
                        kwargs_default,
                        kwargs_theory,
                        i=j,
                    )
                    ax_m.plot(
                        my_data[obs_m].query(f"x=={x}")["A"],
                        my_data[obs_m].query(f"x=={x}")["central"],
                        **kwargs,
                    )

                    if plot_unc:
                        kwargs_uncertainty_default = {
                            "color": col,
                            "alpha": 0.3,
                        }
                        kwargs_u = update_kwargs(
                            kwargs_uncertainty_default,
                            kwargs_uncertainty,
                            i=j,
                        )
                        ax_m.fill_between(
                            my_data[obs_m].query(f"x=={x}")["A"],
                            my_data[obs_m].query(f"x=={x}")["unc1"],
                            my_data[obs_m].query(f"x=={x}")["unc2"],
                            **kwargs_u,
                        )

            else:
                for j, (x, col) in enumerate(zip(x_vals, colors)):
                    kwargs_default = {
                        "color": col,
                        "label": f"x={x}",
                        "linewidth": 1.5,
                    }
                    kwargs = update_kwargs(
                        kwargs_default,
                        kwargs_theory,
                        i=j,
                    )
                    ax_m.plot(
                        my_data[obs_m].query(f"x=={x}")["A"],
                        np.array(my_data[obs_m].query(f"x=={x}")["central"])
                        / np.array(my_data[obs_m].query(f"A=={1} & x=={x}")["central"]),
                        **kwargs,
                    )

                    if plot_unc:
                        kwargs_uncertainty_default = {
                            "color": col,
                            "alpha": 0.3,
                        }
                        kwargs_u = update_kwargs(
                            kwargs_uncertainty_default,
                            kwargs_uncertainty,
                            i=j,
                        )
                        ax_m.fill_between(
                            my_data[obs_m].query(f"x=={x}")["A"],
                            my_data[obs_m].query(f"x=={x}")["unc1"]/ np.array(my_data[obs_m].query(f"A=={1} & x=={x}")["central"]),
                            my_data[obs_m].query(f"x=={x}")["unc2"]/ np.array(my_data[obs_m].query(f"A=={1} & x=={x}")["central"]),
                            **kwargs_u,
                        )
                        
            kwargs_xlabel_default = {
                "fontsize": 14,
                "xlabel": "$A$",
            }
            kwargs_x = update_kwargs(
                kwargs_xlabel_default,
                kwargs_xlabel,
            )

            ax_m.set_xlabel(**kwargs_x)

            kwargs_legend_default = {
                "fontsize": 12,
                "loc": "upper left",
                "bbox_to_anchor": (1, 1),
                "frameon": False,
            }
            kwargs_legend = update_kwargs(
                kwargs_legend_default,
                kwargs_legend,
            )

            if m == len(ax.flat) - 1:
                ax_m.legend(**kwargs_legend)

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
                kwargs_n = update_kwargs(kwargs_notation_default, kwargs_notation, i=m)
                ax_m.annotate(f"${util.to_str(obs_m, Q=Q)}$", **kwargs_n)

            if pdf_label == "ylabel":
                kwargs_ylabel_default = {
                    "fontsize": 14,
                    "ylabel": f"${util.to_str(obs_m, Q=Q)}$",
                }
                kwargs_y = update_kwargs(kwargs_ylabel_default, kwargs_ylabel, i=m)
                ax_m.set_ylabel(**kwargs_y)

            if logx:
                ax_m.set_xscale("log")

        if title:
            kwargs_title_default = {
                "fontsize": 20,
                "y": 1.05,
                "loc": "center"
            }
            kwargs_title = update_kwargs(
                kwargs_title_default,
                kwargs_title,
            )

            if isinstance(title, list):
                for k, title_k in enumerate(title):
                    ax.flatten()[k].set_title(f"{title_k}", **kwargs_title)
            else:
                ax.flatten()[0].set_title(f"{title}", **kwargs_title)
