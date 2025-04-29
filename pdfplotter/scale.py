from __future__ import annotations

from typing_extensions import Literal, Sequence

import numpy as np
import numpy.typing as npt
from matplotlib import axis as maxis
from matplotlib import scale as mscale
from matplotlib import ticker as mticker
from matplotlib import transforms as mtransforms


def map_interval(
    a: tuple[float, float], b: tuple[float, float], x: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    return b[0] + (b[1] - b[0]) / (a[1] - a[0]) * (x - a[0])


def section_masks(
    thresholds: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
) -> list[npt.NDArray[np.bool]]:
    masks = []
    for i in range(len(thresholds) - 1):
        mask = (values >= thresholds[i]) & (values < thresholds[i + 1])
        masks.append(mask)
    return masks


class LogLinScale(mscale.ScaleBase):

    name = "loglin"
    threshold: float
    limits: tuple[float, float]
    base: float
    subs: Sequence[float] | None

    _transform: mtransforms.Transform

    def __init__(
        self,
        axis: maxis.Axis,
        *,
        threshold: float = 0.1,
        threshold_fraction: float = 0.5,
        limits: tuple[float, float] = (0, 1),
        base: float = 10,
        subs: Sequence[float] | None = None,
        nonpositive: Literal["clip", "mask"] = "clip",
    ) -> None:
        super().__init__(axis)
        self.threshold = threshold
        self.threshold_fraction = threshold_fraction
        self.limits = limits
        self.base = base
        self.subs = subs
        self._transform = LogLinTransform(
            threshold, threshold_fraction, limits, base, nonpositive=nonpositive
        )

    def get_transform(self) -> mtransforms.Transform:
        return self._transform

    def set_default_locators_and_formatters(self, axis: maxis.Axis):
        endpoints = [self.limits[0], self.threshold, self.limits[1]]

        axis.set_major_locator(
            SectionedLocator(
                endpoints, [mticker.LogLocator(self.base), mticker.AutoLocator()]
            )
        )
        axis.set_minor_locator(
            SectionedLocator(
                endpoints,
                [mticker.LogLocator(self.base, self.subs), mticker.NullLocator()],
            )
        )
        axis.set_major_formatter(
            SectionedFormatter(
                endpoints,
                [
                    mticker.LogFormatterSciNotation(
                        self.base  # pyright: ignore[reportArgumentType]
                    ),
                    mticker.ScalarFormatter(),
                ],
            )
        )
        axis.set_minor_formatter(
            SectionedFormatter(
                endpoints,
                [
                    mticker.LogFormatterSciNotation(
                        self.base,  # pyright: ignore[reportArgumentType]
                        labelOnlyBase=(self.subs is not None),
                    ),
                    mticker.NullFormatter(),
                ],
            )
        )


class LogLinTransform(mtransforms.Transform):
    input_dims = output_dims = 1

    threshold: float
    threshold_transformed: float
    limits: tuple[float, float]
    limits_transformed: tuple[float, float]
    base: float

    log_transform: mscale.LogTransform

    def __init__(
        self,
        threshold: float,
        threshold_fraction: float,
        limits: tuple[float, float],
        base: float,
        nonpositive: Literal["clip", "mask"] = "clip",
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.threshold_fraction = threshold_fraction
        self.limits = limits
        self.base = base

        self.log_transform = mscale.LogTransform(base=base, nonpositive=nonpositive)
        self.threshold_transformed = self._transform_non_affine(np.array([threshold]))[
            0
        ]
        self.limits_transformed = tuple(self._transform_non_affine(np.array(limits)))

    def inverted(self) -> InvertedLogLinTransform:
        return InvertedLogLinTransform(
            self.threshold_transformed,
            self.threshold_fraction,
            self.limits_transformed,
            self.base,
        )

    def transform_non_affine(self, values: npt.ArrayLike) -> npt.NDArray:
        return self._transform_affine(self._transform_non_affine(values))

    def _transform_non_affine(self, values: npt.ArrayLike) -> npt.NDArray:
        values = np.atleast_1d(values).astype(np.float64)
        mask = values <= self.threshold

        values[mask] = self.log_transform.transform_non_affine(values[mask])

        return values

    def _transform_affine(self, values: npt.ArrayLike) -> npt.NDArray:
        values = np.atleast_1d(values).astype(np.float64)
        mask = values <= self.threshold_transformed

        values[mask] = map_interval(
            (self.limits_transformed[0], self.threshold_transformed),
            (0, 2 * self.threshold_fraction),
            values[mask],
        )
        values[~mask] = map_interval(
            (self.threshold, self.limits_transformed[1]),
            (2 * self.threshold_fraction, 2),
            values[~mask],
        )

        return values


class InvertedLogLinTransform(mtransforms.Transform):
    input_dims = output_dims = 1

    threshold: float
    treshold_fraction: float
    threshold_transformed: float
    limits: tuple[float, float]
    limits_transformed: tuple[float, float]
    base: float

    inverted_log_transform: mscale.InvertedLogTransform

    def __init__(
        self,
        threshold: float,
        threshold_fraction: float,
        limits: tuple[float, float],
        base: float,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.threshold_fraction = threshold_fraction
        self.limits = limits
        self.base = base
        self.inverted_log_transform = mscale.InvertedLogTransform(base=base)
        self.threshold_transformed = self._transform_non_affine(np.array([threshold]))[
            0
        ]
        self.limits_transformed = tuple(self._transform_non_affine(np.array(limits)))

    def transform_non_affine(self, values: npt.ArrayLike) -> npt.NDArray:
        return self._transform_non_affine(self._transform_affine(values))

    def _transform_non_affine(self, values: npt.ArrayLike) -> npt.NDArray:
        values = np.atleast_1d(values).astype(np.float64)
        mask = values <= self.threshold

        values[mask] = self.inverted_log_transform.transform_non_affine(values[mask])

        return values

    def _transform_affine(self, values: npt.ArrayLike) -> npt.NDArray:
        values = np.atleast_1d(values).astype(np.float64)
        mask = values <= 1

        values[mask] = map_interval(
            (0, 2 * self.threshold_fraction),
            (self.limits[0], self.threshold),
            values[mask],
        )
        values[~mask] = map_interval(
            (2 * self.threshold_fraction, 2),
            (self.threshold_transformed, self.limits[1]),
            values[~mask],
        )

        return values

    def inverted(self) -> LogLinTransform:
        return LogLinTransform(
            self.threshold_transformed,
            self.threshold_fraction,
            self.limits_transformed,
            self.base,
        )


class LogLinLocator(mticker.Locator):

    threshold: float
    log_locator_major: mticker.Locator
    log_locator_minor: mticker.Locator
    lin_locator_major: mticker.Locator
    lin_locator_minor: mticker.Locator

    def __init__(
        self,
        threshold: float,
        base: float,
        subs: str | Sequence[int] | None = None,
    ) -> None:
        self.threshold = threshold
        self.log_locator_major = mticker.LogLocator(base)
        self.log_locator_minor = mticker.LogLocator(base, subs)
        self.lin_locator_major = mticker.AutoLocator()
        self.lin_locator_minor = mticker.NullLocator()

    def set_params(
        self,
        threshold: float | None = None,
        base: float | None = None,
        subs: str | Sequence[int] | None = None,
        **kwargs,
    ) -> None:
        if threshold is not None:
            self.threshold = threshold

        self.log_locator_major.set_params(base=base, subs=subs)

    def __call__(self) -> npt.NDArray[np.floating]:
        vlim: tuple[float, float] = self.axis.get_view_interval()
        return self.tick_values(vlim[0], vlim[1])

    def tick_values(self, vmin: float, vmax: float) -> npt.NDArray[np.floating]:
        if vmax < self.threshold:
            return self.lin_locator_major.tick_values(vmin, vmax)
        elif vmin > self.threshold:
            return self.log_locator_major.tick_values(vmin, vmax)
        else:
            t1 = self.log_locator_major.tick_values(vmin, self.threshold)
            t2 = self.lin_locator_major.tick_values(self.threshold, vmax)
            return np.concatenate(
                (
                    t1[t1 < self.threshold],
                    t2[t2 >= self.threshold],
                )
            )

    def view_limits(self, vmin: float, vmax: float) -> tuple[float, float]:
        if vmax < self.threshold:
            return self.lin_locator_major.view_limits(vmin, vmax)
        elif vmin > self.threshold:
            return self.log_locator_major.view_limits(vmin, vmax)
        else:
            return (
                self.log_locator_major.view_limits(vmin, self.threshold)[0],
                self.lin_locator_major.view_limits(self.threshold, vmax)[1],
            )

    def nonsingular(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, vmin: float, vmax: float
    ) -> tuple[float, float]:
        if vmax < self.threshold:
            return self.lin_locator_major.nonsingular(vmin, vmax)
        elif vmin > self.threshold:
            return self.log_locator_major.nonsingular(vmin, vmax)
        else:
            return (
                self.log_locator_major.nonsingular(vmin, self.threshold)[0],
                self.lin_locator_major.nonsingular(self.threshold, vmax)[1],
            )

    def set_axis(self, axis: maxis.Axis) -> None:
        super().set_axis(axis)
        self.log_locator_major.set_axis(axis)
        self.lin_locator_major.set_axis(axis)

    def create_dummy_axis(self, **kwargs):
        if self.axis is None:
            self.set_axis(
                mticker._DummyAxis(**kwargs)  # pyright: ignore[reportArgumentType]
            )


class SectionedLocator(mticker.Locator):
    endpoints: npt.NDArray[np.float64]
    locators: npt.NDArray[mticker.Locator]  # pyright: ignore[reportInvalidTypeForm]

    def __init__(
        self,
        endpoints: Sequence[float],
        locators: Sequence[mticker.Locator],
    ) -> None:
        self.set_params(endpoints, locators)

    def set_params(
        self,
        endpoints: Sequence[float] | None = None,
        locators: Sequence[mticker.Locator] | None = None,
        **kwargs,
    ) -> None:
        if endpoints is not None:
            self.endpoints = np.atleast_1d(endpoints)
        if locators is not None:
            self.locators = np.atleast_1d(
                locators  # pyright: ignore[reportCallIssue,reportArgumentType]
            )

    def set_axis(self, axis: maxis.Axis) -> None:
        super().set_axis(axis)
        for locator in self.locators:
            locator: mticker.Locator
            locator.set_axis(axis)

    def create_dummy_axis(self, **kwargs):
        if self.axis is None:
            self.set_axis(
                mticker._DummyAxis(**kwargs)  # pyright: ignore[reportArgumentType]
            )

    def __call__(self) -> npt.NDArray[np.floating]:
        vlim: tuple[float, float] = self.axis.get_view_interval()
        return self.tick_values(vlim[0], vlim[1])

    def tick_values(self, vmin: float, vmax: float) -> npt.NDArray[np.floating]:
        # indices where vmin and vmax are located in the enpoints array (clip so that the first and last element in self.thresholds would be replaced instead of insertion)
        i = np.clip(
            np.searchsorted(self.endpoints, [vmin, vmax], side="right"),
            1,
            len(self.endpoints) - 1,
        )
        # sections clipped to vmin and vmax
        thresholds_v: npt.NDArray[np.floating] = np.unique(
            np.concat(([vmin], self.endpoints[i[0] : i[1]], [vmax]))
        )

        res = []
        # iterate over the clipped endpoints and their corresponding locators
        for t, l in zip(
            np.lib.stride_tricks.sliding_window_view(thresholds_v, 2),
            self.locators[i[0] - 1 : i[1]],
        ):
            res.append(l.tick_values(t[0], t[1]))

        return np.concat(res) if res else np.array([])  # TODO: is this correct?

    def view_limits(self, vmin: float, vmax: float) -> tuple[float, float]:
        # indices where vmin and vmax are located in the thresholds array (clip so that the first and last element in self.thresholds would be replaced instead of insertion)
        i = np.clip(
            np.searchsorted(self.endpoints, [vmin, vmax], side="right"),
            1,
            len(self.endpoints) - 1,
        )

        # fmt: off
        return (
            self.locators[i[0] - 1].view_limits(vmin, self.endpoints[i[0]])[0],  # pyright: ignore[reportAttributeAccessIssue]
            self.locators[i[1] - 1].view_limits(self.endpoints[i[1] - 1], vmax)[1],  # pyright: ignore[reportAttributeAccessIssue]
        )
        # fmt: on

    def nonsingular(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, vmin: float, vmax: float
    ) -> tuple[float, float]:
        i = np.clip(
            np.searchsorted(self.endpoints, [vmin, vmax], side="right"),
            1,
            len(self.endpoints) - 1,
        )

        # fmt: off
        return (
            self.locators[i[0] - 1].nonsingular(vmin, self.endpoints[i[0]])[0],  # pyright: ignore[reportAttributeAccessIssue]
            self.locators[i[1] - 1].nonsingular(self.endpoints[i[1] - 1], vmax)[1],  # pyright: ignore[reportAttributeAccessIssue]
        )
        # fmt: on


class LogLinFormatter(mticker.Formatter):
    threshold: float
    log_formatter: mticker.LogFormatter
    lin_formatter: mticker.ScalarFormatter

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.log_formatter = mticker.LogFormatterSciNotation()
        self.lin_formatter = mticker.ScalarFormatter()

    def set_axis(self, axis: maxis.Axis) -> None:
        super().set_axis(axis)
        self.log_formatter.set_axis(axis)
        self.lin_formatter.set_axis(axis)

    def set_locs(self, locs: npt.ArrayLike | None = None) -> None:
        if locs is not None:
            locs = np.atleast_1d(locs)
            mask = locs < self.threshold
            locs1 = locs[mask]
            locs2 = locs[~mask]
        else:
            locs1 = None
            locs2 = None

        self.log_formatter.set_locs(locs1)  # pyright: ignore[reportArgumentType]
        self.lin_formatter.set_locs(locs2)  # pyright: ignore[reportArgumentType]

    def __call__(self, x: float, pos: int | None = None):
        if x < self.threshold:
            return self.log_formatter(x, pos=pos)  # pyright: ignore[reportArgumentType]
        else:
            return self.lin_formatter(x, pos=pos)  # pyright: ignore[reportArgumentType]

    def format_data(self, value: float) -> str:
        if value < self.threshold:
            return self.log_formatter.format_data(value)
        else:
            return self.lin_formatter.format_data(value)

    def format_data_short(self, value: float) -> str:
        if value < self.threshold:
            return self.log_formatter.format_data_short(value)
        else:
            return self.lin_formatter.format_data_short(value)


class SectionedFormatter(mticker.Formatter):
    endpoints: npt.NDArray[np.float64]
    formatters: npt.NDArray[mticker.Formatter]  # pyright: ignore[reportInvalidTypeForm]

    def __init__(
        self,
        endpoints: Sequence[float] | None = None,
        formatters: Sequence[mticker.Formatter] | None = None,
    ) -> None:
        if endpoints is not None:
            self.endpoints = np.atleast_1d(endpoints)
        if formatters is not None:
            self.formatters = np.atleast_1d(
                formatters  # pyright: ignore[reportCallIssue,reportArgumentType]
            )

    def set_axis(self, axis: maxis.Axis) -> None:
        super().set_axis(axis)
        for formatter in self.formatters:
            formatter: mticker.Locator
            formatter.set_axis(axis)

    def create_dummy_axis(self, **kwargs):
        if self.axis is None:
            self.set_axis(
                mticker._DummyAxis(**kwargs)  # pyright: ignore[reportArgumentType]
            )

    def set_locs(self, locs: npt.ArrayLike | None = None) -> None:
        if locs is not None:
            locs = np.atleast_1d(locs)
            for mask, formatter in zip(
                section_masks(self.endpoints, locs), self.formatters
            ):
                formatter: mticker.Formatter
                if hasattr(formatter, "set_locs"):
                    formatter.set_locs(locs[mask])
        else:
            for formatter in self.formatters:
                formatter.set_locs(None)  # pyright: ignore[reportArgumentType]

    def __call__(self, x: float, pos: int | None = None):
        i = np.clip(
            np.searchsorted(self.endpoints, [x], side="right"),
            1,
            len(self.endpoints) - 1,
        )

        return self.formatters[i[0] - 1](x, pos=pos)  # pyright: ignore[reportCallIssue]

    def format_data(self, value: float) -> str:
        i = np.clip(
            np.searchsorted(self.endpoints, value, side="right"),
            1,
            len(self.endpoints) - 1,
        )

        return self.formatters[i - 1].format_data(value)

    def format_data_short(self, value: float) -> str:
        i = np.clip(
            np.searchsorted(self.endpoints, value, side="right"),
            1,
            len(self.endpoints) - 1,
        )

        return self.formatters[i - 1].format_data_short(value)


mscale.register_scale(LogLinScale)
