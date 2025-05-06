from __future__ import annotations

from itertools import zip_longest, cycle
from math import log
from typing import Sequence, Literal, Any

#from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.ticker as mticker
import matplotlib.cm as cm
import matplotlib.colors as cls
import math
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp


from pdfplotter.pdf_set import PDFSet
from pdfplotter.util import update_kwargs
from pdfplotter.util import log_tick_formatter
from pdfplotter import elements
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

    def plot_A_dep_2d_scatter(
        self,
        ax: plt.Axes | npt.NDArray[plt.Axes],  # pyright: ignore[reportInvalidTypeForm]
        x: float | list[float],
        observables: (
            sp.Basic
            | npt.NDArray[sp.Basic]  # pyright: ignore[reportInvalidTypeForm]
            | list[sp.Basic]
        ),
        Q: float | None = None,
        Q2: float | None = None,
        A_lines: float | list[float] | None = None,
        colors: list[str] | str | cycle = [],
        offset: float = 0,
        labels_Bjx: Literal["lines", "legend", "none"] = "legend",
        name: str = "",
        plot_ratio: bool = False,
        pdf_label: Literal["title", "annotate", "none"] | None = "annotate",
        plot_legend: bool = True,
        kwargs_theory: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_legend: dict[str, Any] = {},
        kwargs_ylabel: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_title: dict[str, Any] | list[dict[str, Any] | None]= {},
        kwargs_annotate: dict[str, Any] | list[dict[str, Any] | None] = {},
        kwargs_xticks: list[dict[str, Any] | None] = {},
    ) -> None:
        """Plot nuclear PDFs in the A-f plane for different values of x.

        Parameters
        ----------
        ax : matplotlib.axes.Axes | numpy.ndarray[matplotlib.axes.Axes]
            The axes to plot on.
        x : float | list[float]
            The x values to plot for.
        observables : sympy.Basic | numpy.ndarray[sympy.Basic] | list[sympy.Basic]
            The observables to plot.
        Q : float, optional
            The scale at which to plot the PDFs
        Q2 : float, optional
            The Q^2 scale at which to plot the PDFs. Either Q or Q2 has to be passed.
        colors : list, optional
            The colors to use for the different x values, by default [], tab color palette is used if == [].
        logx : bool, optional
            If True, use a logarithmic scale for the x axis, by default True.
        title : str | list[str] | None, optional
            The title of the plot, by default None. If a list is passed, the titles are set for each subplot. If a single string is passed, it is set for the first subplot.
        plot_unc : bool, optional
            If True, plot the uncertainties, by default False.
        plot_ratio : bool, optional
            If True, plot the ratio of the PDFs to the Proon PDF, by default False.
        pdf_label : str, optional
            The label for the PDF, by default "annotate". If "ylabel", the label is set as the y-axis label. If "annotate", the label is set as an anannotate in the top right corner of the plot.
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
        kwargs_annotate : dict[str, Any] | list[dict[str, Any] | None], optional
            The keyword arguments to pass to the anannotate function, by default {}.
        kwargs_uncertainty : dict[str, Any] | list[dict[str, Any] | None], optional
            The keyword arguments to pass to the fill_between function for the uncertainties, by default {}.
        kwargs_uncertainty_edges : dict[str, Any] | list[dict[str, Any] | None], optional
            The keyword arguments to pass to the plot function for the uncertainty edges, by default {}.
        """

        my_sets = self.pdf_sets
        my_data = {}

        if not isinstance(x, list):
            x = [x]

        for x_i in x:
            if x_i not in self.get(A=my_sets["A"][0]).x_values:
                raise ValueError(
                    f"Chosen x value {x_i} was not used for defining nuclear pdf set. \n Pleas choose x that was used in initialization"
                )

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
        if isinstance(observables, np.ndarray):
            observables = list(observables.flatten())

        if not isinstance(observables, list):
            observables = [observables]

        if isinstance(colors, str):
            colors = len(x) * [colors]

        elif isinstance(colors, list) and colors != []:
            if len(colors) != len(x):
                raise ValueError("No. of colors must match no. of x-values")

        for obs in observables:
            data_obs = {}
            list_x = []
            list_central = []
            list_unc1 = []
            list_unc2 = []
            list_A = []
            for A in list(my_sets["A"]):

                for x_i in self.get(A=A).x_values:
                    list_x.append(x_i)
                    list_central.append(
                        self.get(A=A).get_central(x=x_i, Q=Q, Q2=Q2, observable=obs)
                    )
                    unc1 = self.get(A=A).get_uncertainties(
                        x=x_i, Q=Q, Q2=Q2, observable=obs
                    )[0]
                    unc2 = self.get(A=A).get_uncertainties(
                        x=x_i, Q=Q, Q2=Q2, observable=obs
                    )[1]
                    if math.isnan(unc1):
                        list_unc1.append(
                            self.get(A=A).get_central(x=x_i, Q=Q, Q2=Q2, observable=obs)
                        )
                    else:
                        list_unc1.append(abs(self.get(A=A).get_central(x=x_i, Q=Q, Q2=Q2, observable=obs)-unc1))
                    if math.isnan(unc2):
                        list_unc2.append(
                            self.get(A=A).get_central(x=x_i, Q=Q, Q2=Q2, observable=obs)
                        )
                    else:
                        list_unc2.append(abs(-self.get(A=A).get_central(x=x_i, Q=Q, Q2=Q2, observable=obs)+unc2))
                i = 0
                while i < len(self.get(A=A).x_values):
                    list_A.append(A)
                    i += 1

            data_obs["A"] = list_A
            data_obs["x"] = list_x
            data_obs["central"] = list_central
            data_obs["err_mi"] = list_unc1
            data_obs["err_pl"] = list_unc2

            dataframe_obs = pd.DataFrame(data_obs)
            my_data[obs] = dataframe_obs

        # fig, ax = plt.subplots(1, len(observables), figsize=(9 * len(observables), 5))

        if not isinstance(ax, np.ndarray):
            ax = np.array([ax])

        for m, (obs_m, ax_m) in enumerate(zip(observables, ax.flat)):
            ax_m: plt.Axes


            if colors == []:
                colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

            markers=cycle(["x","*","o","^"])

            
            for j, x_j in enumerate(x):
                if isinstance(colors, str):
                    col = colors
                elif isinstance(colors, list):
                    col = colors[j]
                else:
                    col = next(colors)
                mk=next(markers)
                if j==0 and labels_Bjx=="legend":
                    kwargs_default = {
                        "color": col,
                        "label": f"x={x_j}, {name}",
                        "linestyle":"",
                        "capsize":5,
                        "fmt":mk,
                        "ecolor":col
                    }
                elif j!=0 and labels_Bjx=="legend":
                    kwargs_default = {
                        "color": col,
                        "label": f"x={x_j}",
                        "linestyle":"",
                        "capsize":5,
                        "fmt":mk,
                        "ecolor":col
                    }
                elif j==0 and labels_Bjx!="legend":
                    kwargs_default = {
                        "color": col,
                        "label": f"{name}",
                        "linestyle":"",
                        "capsize":5,
                        "fmt":mk,
                        "ecolor":col
                    }
                else:
                    kwargs_default = {
                        "color": col,
                        "linestyle":"",
                        "capsize":5,
                        "fmt":mk,
                        "ecolor":col
                    }                    
                
                kwargs = update_kwargs(
                    kwargs_default,
                    kwargs_theory,
                    i=j,
                )
                if not plot_ratio:
                    ax_m.errorbar(
                        np.array(range(len(A_lines)))+offset*np.ones(shape=len(A_lines)),
                        my_data[obs_m].query(f"x=={x_j} and A in {A_lines}")["central"],
                        yerr=(
                        my_data[obs_m].query(f"x=={x_j} and A in {A_lines}")["err_mi"],
                        my_data[obs_m].query(f"x=={x_j} and A in {A_lines}")["err_pl"]),
                        **kwargs,
                    )
                else:
                    ax_m.errorbar(
                        np.array(range(len(my_data[obs_m].query(f"x=={x_j}and A in {A_lines}")["A"])))+offset*np.ones(shape=len(A_lines)),
                        np.array(list(my_data[obs_m].query(f"x=={x_j} and A in {A_lines}")["central"]))/np.array(my_data[obs_m].query(f"A=={1} & x=={x_j}")["central"]),
                        yerr=(
                        np.array(list(my_data[obs_m].query(f"x=={x_j} and A in {A_lines}")["err_mi"]))/np.array(my_data[obs_m].query(f"A=={1} & x=={x_j}")["central"]),
                        np.array(list(my_data[obs_m].query(f"x=={x_j} and A in {A_lines}")["err_pl"]))/np.array(my_data[obs_m].query(f"A=={1} & x=={x_j}")["central"])),
                        **kwargs,
                    )

            
            if A_lines is not None:
                if not isinstance(A_lines, list):
                    A_lines = [A_lines]

                ax_m.set_xticks(
                    range(len(A_lines)),
                    labels=[
                        f"{elements.element_to_str(A=A_line,long=True)} {A_line}"
                        for A_line in A_lines
                    ],
                    ha="right",
                    rotation=30,
                )
                ax_m.xaxis.set_tick_params(which="minor", size=0)

            ax_m.grid(visible=True)


            kwargs_ylabel_default = {
                "ylabel": f"${util.to_str(obs_m)}$",
            }
            
            if isinstance(kwargs_ylabel, list):
                kwargs_y = update_kwargs(kwargs_ylabel_default, kwargs_ylabel, i=m)
            else:
                kwargs_y = update_kwargs(
                    kwargs_ylabel_default,
                    kwargs_ylabel,
                )

            ax_m.set_ylabel(**kwargs_y)

            if labels_Bjx == "lines":
                if m == len(ax.flat) - 1:

                    kwargs_legend_default = {
                            "loc": "upper left",
                            "bbox_to_anchor": (1, 1),
                            "frameon": False,
                    }
                    kwargs_legend = update_kwargs(
                            kwargs_legend_default,
                            kwargs_legend,
                    )

                    ax_m.legend(**kwargs_legend)
                
                for x_j in x: 
                    if not plot_ratio:
                        ax_m.annotate(f"x={x_j}",fontsize= 10,
                        xy= (0, my_data[obs_m].query(f"x=={x_j} and A ==1 ")["central"].iloc[0]))  
            if labels_Bjx == "legend":
                if plot_legend:
                    kwargs_legend_default = {
                            "loc": "upper left",
                            "bbox_to_anchor": (1, 1),
                            "frameon": False,
                        }
                    kwargs_legend = update_kwargs(
                            kwargs_legend_default,
                            kwargs_legend,
                        )

                    ax[-1].legend(**kwargs_legend)

            if pdf_label == "annotate":
                kwargs_annotate_default = {
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
                kwargs_n = update_kwargs(kwargs_annotate_default, kwargs_annotate, i=m)
                ax_m.annotate(f"${util.to_str(obs_m, Q=Q,Q2=Q2)}$", **kwargs_n)

            if pdf_label == "title":
                kwargs_title_default = {
                "y": 1.05,
                "loc": "center",
                "label": f"${util.to_str(obs_m,Q=Q,Q2=Q2)}$",
                }
                if isinstance(kwargs_title, list):
                    kwargs_t = update_kwargs(kwargs_title_default, kwargs_title, i=m)
                else:
                    kwargs_t = update_kwargs(kwargs_title_default, kwargs_title)
                ax_m.set_title(**kwargs_t)
