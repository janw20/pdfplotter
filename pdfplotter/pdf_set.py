from __future__ import annotations

import lhapdf
import numpy as np
import numpy.typing as npt
import pandas as pd
import sympy as sp
from matplotlib import pyplot as plt
from typing_extensions import Any, Literal, Sequence

from pdfplotter.flavors import flavors_nucleus, isospin_transform, pid_from_flavor
from pdfplotter.util import update_kwargs

idx = pd.IndexSlice


class PDFSet:
    """Wrapper of the LHAPDF PDFSet class, including the calculation of observables, ratios, and their uncertainties."""

    _name: str
    _pdf_set: lhapdf.PDFSet
    _x: np.floating | npt.NDArray[np.floating]
    _Q: np.floating | npt.NDArray[np.floating]
    _Q2: np.floating | npt.NDArray[np.floating]
    _pdfs: list[lhapdf.PDF]
    _data: pd.DataFrame
    """DataFrame with index `(Q, member, x) ∈ Q_values × {0, ..., len(_pdfs)} × x_values` and columns `flavor ∈ {"d", "d̅", "d bound", ...}`"""
    _observables: pd.DataFrame
    """DataFrame with index `(pdf_type, Q, x) ∈ {"central", "unc+", "unc-"} × Q_values × x_values` and columns `observable ∈ {"u/d", "u/d bound", "u_v", ...}`"""
    _ratios: pd.DataFrame
    """DataFrame with index `(pdf_type, Q, x) ∈ {"central", "unc+", "unc-"} × Q_values × x_values` and columns `observable ∈ {"u/d <denominator PDFSet name>", ...}`"""
    _A: int
    _Z: int
    _construct_full_nuclear_pdfs: bool
    _confidence_level: float
    _uncertainty_type: str
    _num_errors: int

    def __init__(
        self,
        name: str,
        x: float | Sequence[float] | npt.NDArray[np.floating],
        Q: float | Sequence[float] | npt.NDArray[np.floating] | None = None,
        Q2: float | Sequence[float] | npt.NDArray[np.floating] | None = None,
        A=1,
        Z=1,
        construct_full_nuclear_pdfs=False,
        confidence_level: float = 90,
    ) -> None:
        """Constructs a PDFSet object.

        Parameters
        ----------
        name : str
            The name of the LHAPDF set to load.
        x : numpy.ArrayLike
            The momentum fraction values at which the PDFs are evaluated.
        Q : numpy.ArrayLike, optional
            The scale values at which the PDFs are evaluated. Either `Q` or `Q2` must be given.
        Q2 : numpy.ArrayLike, optional
            The Q² (squared scale) values at which the PDFs are evaluated. Either `Q` or `Q2` must be given.
        A : int, optional
            Mass number of the nucleus for nuclear PDFs, by default 1 (proton PDFs)
        Z : int, optional
            Charge number of the nucleus for nuclear PDFs, by default 1 (proton PDFs)
        construct_full_nuclear_pdfs : bool, optional
            True if full nuclear PDFs should be constructed using `xf^A = Z/A * xf^(p/A) + (A - Z)/A * xf^(n/A). Only use this if the LHAPDF set you are loading does not already load full nuclear PDFs. By default False.
        confidence_level : float, optional
            The confidence level in percent at which the uncertainties are calculated. By default 90.
        """
        if Q is None:
            if Q2 is None:
                raise ValueError("Either Q or Q2 must be given")
            self._Q2 = np.array(Q2)[()]
            self._Q = np.sqrt(Q2)
        elif Q2 is None:
            self._Q = np.array(Q)[()]
            self._Q2 = self._Q**2
        elif Q is not None and Q2 is not None:
            raise ValueError("Only one of Q or Q2 must be given")

        self._x = np.array(x)[()]

        if Z > A:
            raise ValueError("Z must be less than or equal to A")
        self._name = name

        # need to check ourselves if the PDF set is available, otherwise the notebook kernel crashes
        if not name in lhapdf.availablePDFSets():
            raise FileNotFoundError(
                f"PDF set {name} was not found or has missing info file"
            )

        self._pdf_set = lhapdf.getPDFSet(self.name)
        self._pdfs = self.pdf_set.mkPDFs()
        self._num_errors = self.pdf_set.errSize
        self._A = A
        self._Z = Z
        self._construct_full_nuclear_pdfs = construct_full_nuclear_pdfs
        self._confidence_level = (
            confidence_level if confidence_level else self.pdf_set.errorConfLevel
        )
        self._uncertainty_type = self.pdf_set.errorType

        Q_values = np.atleast_1d(self._Q)
        x_values = np.atleast_1d(self._x)
        self._data = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [Q_values, np.arange(self._pdf_set.size), x_values],
                names=["Q", "member", "x"],
            )
        ).sort_index()
        pdf_types = ["central", "unc_asym_+", "unc_asym_-", "unc_sym_+", "unc_sym_-"]
        self._observables = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [pdf_types, Q_values, x_values], names=["pdf_type", "Q", "x"]
            ).sort_values()
        )
        self._ratios = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [pdf_types, Q_values, x_values], names=["pdf_type", "Q", "x"]
            ).sort_values()
        )

    @property
    def name(self) -> str:
        """The name of the LHAPDF set."""
        return self._name

    @property
    def pdf_set(self) -> lhapdf.PDFSet:
        """The LHAPDF set."""
        return self._pdf_set

    @property
    def pdfs(self) -> list[lhapdf.PDF]:
        """The PDFs in the LHAPDF set."""
        return self._pdfs

    @property
    def A(self) -> int:
        """The mass number of the nucleus. 1 for proton PDFs."""
        return self._A

    @property
    def Z(self) -> int:
        """The charge number of the nucleus. 1 for proton PDFs."""
        return self._Z

    @property
    def has_full_nuclear_pdfs(self) -> bool:
        """True if the LHAPDF set contains full nuclear PDFs, False if it contains proton or bound proton PDFs."""
        return self._construct_full_nuclear_pdfs

    @property
    def has_bound_proton_pdfs(self) -> bool:
        """True if the LHAPDF set contains bound proton PDFs, False if it contains full nuclear PDFs or proton PDFs."""
        return self.A != 1 and not self.has_full_nuclear_pdfs

    @property
    def confidence_level(self) -> float:
        """The confidence level of the uncertainties."""
        return self._confidence_level

    @property
    def uncertainty_type(self) -> str:
        """The type of the uncertainties."""
        return self._uncertainty_type

    @property
    def num_errors(self) -> int:
        """The number of error PDFs in the PDF set."""
        return self._num_errors

    @property
    def x(self) -> np.floating | npt.NDArray[np.floating]:
        """The momentum fraction values at which the PDFs are evaluated."""
        return self._x

    @property
    def Q(self) -> np.floating | npt.NDArray[np.floating]:
        """The scale values at which the PDFs are evaluated."""
        return self._Q

    @property
    def Q2(self) -> np.floating | npt.NDArray[np.floating]:
        """The squared scale values at which the PDFs are evaluated."""
        return self._Q2

    def _check_and_load_flavors(self, term: sp.Basic) -> None:
        """Checks if the PDF flavors in `term` are already stored in `_data` and loads them if not. Bound proton PDFs are appended by \"bound\".

        Parameters
        ----------
        term : sympy.Basic
            The term to check for PDF flavors.
        """
        new_flavors = {
            flavor
            for flavor in term.free_symbols
            if not str(flavor) in self._data.columns
        }
        new_flavors_nuclear: set[sp.Basic] = set()
        new_flavors_bound_proton: set[sp.Basic] = set()
        if self._construct_full_nuclear_pdfs:
            new_flavors_nuclear = new_flavors & flavors_nucleus
            new_flavors_bound_proton = new_flavors_nuclear.union(
                isospin_transform[f] for f in new_flavors_nuclear
            )
            new_flavors -= new_flavors_nuclear

        for flavor in new_flavors_bound_proton:
            self._data[str(flavor) + " bound"] = [self._pdfs[member].xfxQ(pid_from_flavor[flavor], x, Q) for Q, member, x in self._data.index]  # type: ignore

        # construct nuclear PDFs which are not u, d, u̅, d̅
        for flavor in new_flavors:
            self._data[str(flavor)] = [self._pdfs[member].xfxQ(pid_from_flavor[flavor], x, Q) for Q, member, x in self._data.index]  # type: ignore

        # construct nuclear PDFs
        for flavor in new_flavors_nuclear:
            self._data[str(flavor)] = (
                self.Z / self.A * self._data[str(flavor) + " bound"]
                + (self.A - self.Z)
                / self.A
                * self._data[str(isospin_transform[flavor]) + " bound"]
            )

    def _check_and_calculate_observable(self, term: sp.Basic) -> None:
        """Checks if the observable `term` is already stored in `_observables` and calculates it if not.

        Parameters
        ----------
        term : sympy.Basic
            The observable to calculate.
        """
        self._check_and_load_flavors(term)
        if not str(term) in self._observables.columns:
            # put term.free_symbols in a list since sp.lambdify depends on the order of the symbols
            symbols_ordered = list(term.free_symbols)
            if not str(term) in self._data.columns:
                self._data[str(term)] = sp.lambdify(symbols_ordered, term)(
                    *[self._data[str(flavor)] for flavor in symbols_ordered]
                )

            if len(self.pdfs) == 1:
                self._observables[str(term)] = (
                    self._data.loc[idx[:, 0, :], str(term)]
                    .droplevel(1)
                    .reset_index()
                    .assign(pdf_type="central")
                    .set_index(["pdf_type", "Q", "x"])
                )
            else:
                self._observables[str(term)] = self._data[str(term)].unstack("member").apply(lambda x: self._pdf_set.uncertainty(x, cl=self.confidence_level), axis=1).apply(func=(lambda x: pd.Series([x.central, x.central + x.errplus, x.central - x.errminus, x.central + x.errsymm, x.central - x.errsymm], index=self._observables.index.get_level_values("pdf_type").unique()))).stack().reorder_levels(["pdf_type", "Q", "x"])  # type: ignore

    def _check_and_calculate_ratio(self, term: sp.Basic, denominator: "PDFSet") -> None:
        """Check if the ratio of the observable `term` is already stored in `_ratios` and calculates it if not.

        Parameters
        ----------
        term : sympy.Basic
            The observable to calculate the ratio of.
        denominator : PDFSet, optional
            The PDF set of which the observable `term` in the denominator is taken of, by default None.
        """

        if (
            denominator
            and not str(term) + " " + denominator.name in self._ratios.columns
        ):
            self._check_and_calculate_observable(term)
            denominator._check_and_calculate_observable(term)

            if (
                self.uncertainty_type == "replicas"
                and denominator.uncertainty_type == "replicas"
            ):
                self._ratios[str(term) + " " + denominator.name] = (
                    (self._data[str(term)] / denominator._data[str(term)])
                    .unstack("member")
                    .apply(
                        lambda x: self._pdf_set.uncertainty(
                            x, cl=self.confidence_level
                        ),
                        axis=1,
                    )
                    .apply(
                        func=(
                            lambda x: pd.Series(
                                [
                                    x.central,
                                    x.central + x.errplus,
                                    x.central - x.errminus,
                                    x.central + x.errsymm,
                                    x.central - x.errsymm,
                                ],
                                index=self._ratios.index.get_level_values(
                                    "pdf_type"
                                ).unique(),
                            )
                        )
                    )
                    .stack()
                    .reorder_levels(["pdf_type", "Q", "x"])
                )
            else:
                self._ratios[str(term) + " " + denominator.name] = (
                    self._data[str(term)]
                    .unstack("member")
                    .divide(denominator._data.loc[:, 0, :][str(term)], axis=0)
                    .apply(
                        lambda x: self._pdf_set.uncertainty(
                            x, cl=self.confidence_level
                        ),
                        axis=1,
                    )
                    .apply(
                        func=(
                            lambda x: pd.Series(
                                [
                                    x.central,
                                    x.central + x.errplus,
                                    x.central - x.errminus,
                                    x.central + x.errsymm,
                                    x.central - x.errsymm,
                                ],
                                index=self._ratios.index.get_level_values(
                                    "pdf_type"
                                ).unique(),
                            )
                        )
                    )
                    .stack()
                    .reorder_levels(["pdf_type", "Q", "x"])
                )

    def _flatten_x_Q(
        self,
        x: float | Sequence[float] | npt.NDArray[np.floating] | None = None,
        Q: float | Sequence[float] | npt.NDArray[np.floating] | None = None,
        Q2: float | Sequence[float] | npt.NDArray[np.floating] | None = None,
    ) -> tuple[
        np.floating | npt.NDArray[np.floating] | slice,
        np.floating | npt.NDArray[np.floating] | slice,
        tuple[()] | tuple[int, ...],
    ]:
        x_flat: np.floating | npt.NDArray[np.floating] | slice
        Q_flat: np.floating | npt.NDArray[np.floating] | slice

        if x is None:
            x_flat = idx[:]
            shape_x = self.x.shape
        else:
            x_flat = np.array(x)
            shape_x = x_flat.shape
            x_flat = x_flat.flatten()[()]

        if Q is None and Q2 is None:
            Q_flat = idx[:]
            shape_Q = self.Q.shape
        elif Q is None and Q2 is not None:
            Q_flat = np.sqrt(Q2)
            shape_Q = Q_flat.shape
            Q_flat = Q_flat.flatten()[()]
        elif Q is not None and Q2 is None:
            Q_flat = np.array(Q)
            shape_Q = Q_flat.shape
            Q_flat = Q_flat.flatten()[()]
        else:
            raise ValueError("Only one of Q and Q2 must be given")

        return x_flat, Q_flat, shape_x + shape_Q

    def _reshape(
        self, a: npt.NDArray[np.floating], shape: tuple[int, ...] | None = None
    ) -> npt.NDArray[np.floating] | np.floating:
        return a.reshape(shape if shape is not None else a.shape)[()]

    def get_central(
        self,
        observable: sp.Basic,
        x: npt.ArrayLike | None = None,
        Q: npt.ArrayLike | None = None,
        Q2: npt.ArrayLike | None = None,
        ratio_to: PDFSet | None = None,
    ) -> npt.NDArray[np.floating] | np.floating:
        """Gets the central value of an observable or a ratio of observables.

        Parameters
        ----------
        observable : sympy.Basic
            The observable to get the central value of.
        x : numpy.ArrayLike, optional
            The momentum fraction at which the observable is calculated. By default None, which means that all values in `x_values` are used. If only `x` is given, a 2d-array of shape (`self.Q_values`, `x`) is returned. If both `x` and `Q` (or `Q2`) are given, they are broadcast according to NumPy's broadcasting rules. If none of them are given, a 2d-array of shape (`self.Q_values`, `self.x_values`) is returned.
        Q : numpy.ArrayLike, optional
            The scale at which the observable is calculated. Only one of `Q` or `Q2` must be given. By default None, which means that all values in `Q_values` are used. If only `Q` is given, a 2d-array of shape (`self.x_values`, `Q`) is returned. If both `x` and `Q` are given, they are broadcast according to NumPy's broadcasting rules. If none of them are given, a 2d-array of shape (`self.Q_values`, `self.x_values`) is returned.
        Q2 : numpy.ArrayLike, optional
            The squared scale at which the observable is calculated. Only one of `Q` or `Q2` must be given. By default None, which means that all values in `Q2_values` are used. If only `Q2` is given, a 2d-array of shape (`self.x_values`, `Q2`) is returned. If both `x` and `Q2` are given, they are broadcast according to NumPy's broadcasting rules. If none of them are given, a 2d-array of shape (`self.Q2_values`, `self.x_values`) is returned.
        ratio_to : PDFSet, optional
            For calculating ratios: the PDF set of which the `observable` in the denominator is taken of, by default None, i.e. no ratio.

        Returns
        -------
        numpy.ndarray[np.floating]
            The values of the PDF or the observable.
        """
        x_flat, Q_flat, shape = self._flatten_x_Q(x, Q, Q2)

        if ratio_to is not None:
            self._check_and_calculate_ratio(observable, ratio_to)
            result = self._ratios.loc[idx["central", Q_flat, x_flat], str(observable) + " " + ratio_to.name].to_numpy()  # type: ignore
        else:
            self._check_and_calculate_observable(observable)
            result = self._observables.loc[idx["central", Q_flat, x_flat], str(observable)].to_numpy()  # type: ignore

        return self._reshape(result, shape)

    def get_member(
        self,
        observable: sp.Basic,
        member: int,
        x: float | Sequence[float] | npt.NDArray[np.floating] | None = None,
        Q: float | Sequence[float] | npt.NDArray[np.floating] | None = None,
        Q2: float | Sequence[float] | npt.NDArray[np.floating] | None = None,
    ) -> npt.NDArray[np.floating]:
        x_flat, Q_flat, shape = self._flatten_x_Q(x, Q, Q2)

        self._check_and_calculate_observable(observable)
        return self._reshape(self._data.loc[(Q_flat, member, x_flat), str(observable)].to_numpy(), shape)  # type: ignore

    def get_uncertainty(
        self,
        side: Literal["+", "-"],
        observable: sp.Basic,
        x: float | Sequence[float] | npt.NDArray[np.floating] | None = None,
        Q: float | Sequence[float] | npt.NDArray[np.floating] | None = None,
        Q2: float | Sequence[float] | npt.NDArray[np.floating] | None = None,
        ratio_to: PDFSet | None = None,
        convention: Literal["sym", "asym"] = "sym",
    ) -> npt.NDArray[np.floating] | np.floating:
        """Gets the uncertainties of an observable or a ratio of observables.

        Parameters
        ----------
        side : Literal["+", "-"]
            The side of the uncertainty to get. "+" for the upper uncertainty, "-" for the lower one.
        observable : sympy.Basic
            The observable to get the uncertainties of.
        x : numpy.ArrayLike, optional
            The momentum fraction at which the observable is calculated. By default None, which means that all values in `x_values` are used. If only `x` is given, a 2d-array of shape (`self.Q_values`, `x`) is returned. If both `x` and `Q` (or `Q2`) are given, they are broadcast according to NumPy's broadcasting rules. If none of them are given, a 2d-array of shape (`self.Q_values`, `self.x_values`) is returned.
        Q : numpy.ArrayLike, optional
            The scale at which the observable is calculated. Only one of `Q` or `Q2` must be given. By default None, which means that all values in `Q_values` are used. If only `Q` is given, a 2d-array of shape (`self.x_values`, `Q`) is returned. If both `x` and `Q` are given, they are broadcast according to NumPy's broadcasting rules. If none of them are given, a 2d-array of shape (`self.Q_values`, `self.x_values`) is returned.
        Q2 : numpy.ArrayLike, optional
            The squared scale at which the observable is calculated. Only one of `Q` or `Q2` must be given. By default None, which means that all values in `Q2_values` are used. If only `Q2` is given, a 2d-array of shape (`self.x_values`, `Q2`) is returned. If both `x` and `Q2` are given, they are broadcast according to NumPy's broadcasting rules. If none of them are given, a 2d-array of shape (`self.Q2_values`, `self.x_values`) is returned.
        ratio_to : PDFSet, optional
            For calculating ratios: the PDF set of which the `observable` in the denominator is taken of, by default None, i.e. no ratio.
        convention : "sym" or "asym", optional
            The convention for the uncertainty. "sym" for symmetric uncertainties, "asym" for asymmetric uncertainties. By default "sym".

        Returns
        -------
        numpy.ndarray[np.floating]
            The values of the lower or upper uncertainties.
        """
        x_flat, Q_flat, shape = self._flatten_x_Q(x, Q, Q2)

        if ratio_to:
            self._check_and_calculate_ratio(observable, ratio_to)
            result = self._ratios.loc[idx["_".join(["unc", convention, side]), Q_flat, x_flat], str(observable) + " " + ratio_to.name].to_numpy()  # type: ignore
        else:
            self._check_and_calculate_observable(observable)
            result = self._observables.loc[idx["_".join(["unc", convention, side]), Q_flat, x_flat], str(observable)].to_numpy()  # type: ignore

        return self._reshape(result, shape)

    def get_uncertainties(
        self,
        observable: sp.Basic,
        x: float | Sequence[float] | npt.NDArray[np.floating] | None = None,
        Q: float | Sequence[float] | npt.NDArray[np.floating] | None = None,
        Q2: float | Sequence[float] | npt.NDArray[np.floating] | None = None,
        ratio_to: PDFSet | None = None,
        convention: Literal["sym", "asym"] = "sym",
    ) -> tuple[
        npt.NDArray[np.floating] | np.floating, npt.NDArray[np.floating] | np.floating
    ]:
        """Convenience function to get the lower and upper uncertainties as a tuple, e.g. for use with the unpacking operator.

        Parameters
        ----------
        observable : sympy.Basic
            The observable to get the uncertainties of.
        x : numpy.ArrayLike, optional
            The momentum fraction at which the observable is calculated. By default None, which means that all values in `x_values` are used. If only `x` is given, a 2d-array of shape (`self.Q_values`, `x`) is returned. If both `x` and `Q` (or `Q2`) are given, they are broadcast according to NumPy's broadcasting rules. If none of them are given, a 2d-array of shape (`self.Q_values`, `self.x_values`) is returned.
        Q : numpy.ArrayLike, optional
            The scale at which the observable is calculated. Only one of `Q` or `Q2` must be given. By default None, which means that all values in `Q_values` are used. If only `Q` is given, a 2d-array of shape (`self.x_values`, `Q`) is returned. If both `x` and `Q` are given, they are broadcast according to NumPy's broadcasting rules. If none of them are given, a 2d-array of shape (`self.Q_values`, `self.x_values`) is returned.
        Q2 : numpy.ArrayLike, optional
            The squared scale at which the observable is calculated. Only one of `Q` or `Q2` must be given. By default None, which means that all values in `Q2_values` are used. If only `Q2` is given, a 2d-array of shape (`self.x_values`, `Q2`) is returned. If both `x` and `Q2` are given, they are broadcast according to NumPy's broadcasting rules. If none of them are given, a 2d-array of shape (`self.Q2_values`, `self.x_values`) is returned.
        ratio_to : PDFSet, optional
            For calculating ratios: the PDF set of which the `observable` in the denominator is taken of, by default None, i.e. no ratio.
        convention : "sym" or "asym", optional
            The convention for the uncertainty. "sym" for symmetric uncertainties, "asym" for asymmetric uncertainties. By default "sym".

        Returns
        -------
        numpy.ndarray[np.floating], numpy.ndarray[np.floating]
            The values of the lower and upper uncertainties as a tuple.
        """
        return self.get_uncertainty(
            "+", observable, x, Q, Q2, ratio_to, convention
        ), self.get_uncertainty("-", observable, x, Q, Q2, ratio_to, convention)

    def plot(
        self,
        ax: plt.Axes,
        observable: sp.Basic,
        x: float | Sequence[float] | npt.NDArray[np.floating] | None = None,
        Q: float | Sequence[float] | npt.NDArray[np.floating] | None = None,
        Q2: float | Sequence[float] | npt.NDArray[np.floating] | None = None,
        ratio_to: PDFSet | None = None,
        uncertainty_convention: Literal["sym", "asym"] = "sym",
        variable: Literal["x", "Q", "Q2"] = "x",
        central: bool = True,
        uncertainty: bool = True,
        uncertainty_edges: bool = True,
        kwargs_central: dict[str, Any] = {},
        kwargs_uncertainty: dict[str, Any] = {},
        kwargs_uncertainty_edges: dict[str, Any] = {},
    ) -> None:
        """Plots an observable of this PDF set.

        Parameters
        ----------
        ax : plt.Axes
            The axes to plot on.
        observable : sp.Basic
            The observable to plot.
        x : np.floating | npt.NDArray[np.floating] | None, optional
            The x values to plot, by default None. If None, the values in `self.x` are used.
        Q : np.floating | npt.NDArray[np.floating] | None, optional
            The Q values to plot, by default None. If None and `Q2` is also None, the values in `self.Q` are used.
        Q2 : np.floating | npt.NDArray[np.floating] | None, optional
            The Q2 values to plot, by default None. If None and `Q` is also None, the values in `self.Q2` are used.
        ratio_to : PDFSet | None, optional
            For calculating ratios: the PDF set of which the `observable` in the denominator is taken of, by default None, i.e. no ratio.
        uncertainty_convention : "sym" or "asym", optional
            The convention for the uncertainty. "sym" for symmetric uncertainties, "asym" for asymmetric uncertainties. By default "sym".
        variable : Literal["x", "Q", "Q2"], optional
            The variable on the x axis, by default "x"
        central : bool, optional
            Whether to plot the central value, by default True
        uncertainty : bool, optional
            Whether to plot the uncertainty band, by default True
        uncertainty_edges : bool, optional
            Whether to plot edges around the uncertainty band, by default True
        kwargs_central : dict[str, Any], optional
            Additional keyword arguments for the central PDF that should be passed to `plt.Axes.plot`, by default {}
        kwargs_uncertainty : dict[str, Any], optional
            Additional keyword arguments for the PDF uncertainty band that should be passed to `plt.Axes.fill_between`, by default {}
        kwargs_uncertainty_edges : dict[str, Any], optional
            Additional keyword arguments for the edges of the PDF uncertainty band that should be passed to `plt.Axes.plot`, by default {}
        """
        if variable == "x":
            variable_values = x if x is not None else self.x
        elif variable == "Q":
            variable_values = Q if Q is not None else self.Q
        elif variable == "Q2":
            variable_values = Q2 if Q2 is not None else self.Q2
        else:
            raise ValueError("variable must be either 'x', 'Q' or 'Q2'")

        if central:
            kwargs_default = {}
            kwargs = update_kwargs(kwargs_default, kwargs_central)

            l = ax.plot(
                variable_values,
                self.get_central(
                    observable=observable, x=x, Q=Q, Q2=Q2, ratio_to=ratio_to
                ),
                **kwargs,
            )[0]
        else:
            l = None

        if uncertainty:
            kwargs_from_central = (
                {
                    "facecolor": l.get_color(),
                }
                if l is not None
                else {}
            )
            kwargs_default = {
                "alpha": 0.3,
                "lw": 0,
            } | kwargs_from_central
            kwargs = update_kwargs(kwargs_default, kwargs_uncertainty)

            ax.fill_between(
                variable_values,
                *self.get_uncertainties(
                    observable=observable,
                    x=x,
                    Q=Q,
                    Q2=Q2,
                    ratio_to=ratio_to,
                    convention=uncertainty_convention,
                ),  # pyright: ignore[reportArgumentType]
                **kwargs,
            )

        if uncertainty_edges:
            kwargs_from_central = (
                {
                    "color": l.get_color(),
                    "ls": l.get_linestyle(),
                }
                if l is not None
                else {}
            )
            kwargs_default = {
                "lw": 0.5,
            } | kwargs_from_central
            kwargs = update_kwargs(kwargs_default, kwargs_uncertainty_edges)

            ax.plot(
                variable_values,
                self.get_uncertainty(
                    "+",
                    observable=observable,
                    x=x,
                    Q=Q,
                    Q2=Q2,
                    ratio_to=ratio_to,
                    convention=uncertainty_convention,
                ),
                **kwargs,
            )
            ax.plot(
                variable_values,
                self.get_uncertainty(
                    "-",
                    observable=observable,
                    x=x,
                    Q=Q,
                    Q2=Q2,
                    ratio_to=ratio_to,
                    convention=uncertainty_convention,
                ),
                **kwargs,
            )
