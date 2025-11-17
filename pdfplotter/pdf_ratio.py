from __future__ import annotations

from abc import ABC, abstractmethod
import re
import pandas as pd

import numpy.typing as npt

import numpy as np


class PDFRatioPrescription(ABC):
    """Class to describe how ratios of PDF sets should be calculated."""

    @abstractmethod
    def matches(self, pdf_name: str) -> bool:
        """Determine if the ratio prescription applies to `pdf_name`.

        Parameters
        ----------
        pdf_name : str
            Name of the PDF set to be tested.

        Returns
        -------
        bool
            True if the ratio prescription applies, False otherwise.
        """
        ...

    @abstractmethod
    def calculate_ratio(
        self,
        numerator: npt.NDArray[np.float64],
        denominator: npt.NDArray[np.float64],
        denominator_name: str,
    ) -> npt.NDArray[np.float64]:
        """Calculate the ratio between two PDFs.

        Parameters
        ----------
        numerator : npt.NDArray[np.float64]
            Values of the PDF members of the numerator.
        denominator : npt.NDArray[np.float64]
            Values of the PDF members of the denominator.
        denominator_name : str
            PDF set name of the denominator.

        Returns
        -------
        npt.NDArray[np.float64]
            Values of the ratio of the numerator and the denominator, with the same length as the numerator.
        """


class PDFRatioPrescription_EPPS16(PDFRatioPrescription):

    pattern = re.compile(r"^EPPS16nlo_CT14nlo_[A-Z][a-z][0-9]+$")
    """Regex pattern to match EPPS16 and EPPS21"""

    def matches(self, pdf_name: str) -> bool:
        return re.match(self.pattern, pdf_name) is not None

    def calculate_ratio(
        self,
        numerator: npt.NDArray[np.float64],
        denominator: npt.NDArray[np.float64],
        denominator_name: str,
    ) -> npt.NDArray[np.float64]:
        """Calculate the EPPS ratio according to arXiv:2112.12462v2 eq. 39 & 40."""
        if denominator_name == "CT14nlo":
            return np.concat(
                [
                    numerator[: -denominator.size] / denominator[0],
                    numerator[-denominator.size :] / denominator,
                ]
            )
        else:
            return numerator / denominator[0]


class PDFRatioPrescription_EPPS21(PDFRatioPrescription):

    pattern = re.compile(r"^EPPS21nlo_CT18Anlo_[A-Z][a-z][0-9]+$")
    """Regex pattern to match EPPS16 and EPPS21"""

    def matches(self, pdf_name: str) -> bool:
        return re.match(self.pattern, pdf_name) is not None

    def calculate_ratio(
        self,
        numerator: npt.NDArray[np.float64],
        denominator: npt.NDArray[np.float64],
        denominator_name: str,
    ) -> npt.NDArray[np.float64]:
        """Calculate the EPPS ratio according to arXiv:2112.12462v2 eq. 39 & 40."""
        if denominator_name == "CT18ANLO":
            return np.concat(
                [
                    numerator[: -denominator.size] / denominator[0],
                    numerator[-denominator.size :] / denominator,
                ]
            )
        else:
            return numerator / denominator[0]


class PDFRatioPrescription_nNNPDF30(PDFRatioPrescription):

    pattern = re.compile(r"^nNNPDF30_nlo_as_0118_A[0-9]+_Z[0-9]+$")
    """Regex pattern to match nNNPDF30_nlo_as_0118"""

    def matches(self, pdf_name: str) -> bool:
        return re.match(self.pattern, pdf_name) is not None

    def calculate_ratio(
        self,
        numerator: npt.NDArray[np.float64],
        denominator: npt.NDArray[np.float64],
        denominator_name: str,
    ) -> npt.NDArray[np.float64]:
        """Calculate the nNNPDF3.0 ratio according to arXiv:2201.12363v2 sec. 7.2"""
        if denominator_name == "nNNPDF30_nlo_as_0118_p":
            assert numerator.size == denominator.size

            return numerator / denominator
        else:
            return numerator / denominator[0]


class PDFRatioPrescription_nNNPDF20(PDFRatioPrescription):

    pattern = re.compile(rf"^nNNPDF20_nlo_as_0118_[A-Z]+[a-z]+[0-9]+$")
    """Regex pattern to match nNNPDF20_nlo_as_0118"""

    def matches(self, pdf_name: str) -> bool:
        return re.match(self.pattern, pdf_name) is not None

    def calculate_ratio(
        self,
        numerator: npt.NDArray[np.float64],
        denominator: npt.NDArray[np.float64],
        denominator_name: str,
    ) -> npt.NDArray[np.float64]:
        """Calculate the nNNPDF2.0 ratio according to arXiv:2201.12363v2 sec. 7.2"""
        if denominator_name == "nNNPDF20_nlo_as_0118_N1":
            assert numerator.size == denominator.size

            return numerator / denominator
        else:
            return numerator / denominator[0]


class PDFRatioPrescription_Default(PDFRatioPrescription):

    def matches(self, pdf_name: str) -> bool:
        return True

    def calculate_ratio(
        self,
        numerator: npt.NDArray[np.float64],
        denominator: npt.NDArray[np.float64],
        denominator_name: str,
    ) -> npt.NDArray[np.float64]:
        return numerator / denominator[0]


# DefaultRatioPrescription needs to be the first one
pdf_ratio_prescriptions = [
    PDFRatioPrescription_Default(),
    PDFRatioPrescription_EPPS16(),
    PDFRatioPrescription_EPPS21(),
    PDFRatioPrescription_nNNPDF20(),
    PDFRatioPrescription_nNNPDF30(),
]


def get_ratio_prescription(pdf_name: str) -> PDFRatioPrescription:
    matches = [r for r in pdf_ratio_prescriptions if r.matches(pdf_name)]

    # DefaultRatioPrescription matches everything
    assert len(matches) > 0

    if len(matches) > 2:
        raise RuntimeError(f"Multiple matches {matches[1:]} found for '{pdf_name}'")
    else:
        # return match or DefaultRatioPrescription
        return matches[-1]
