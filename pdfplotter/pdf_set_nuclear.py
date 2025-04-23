from __future__ import annotations

from itertools import zip_longest
from typing import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from pdfplotter.pdf_set import PDFSet


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
