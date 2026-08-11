from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol, runtime_checkable

from typing_extensions import override

from tqec.circuit.schedule.circuit import ScheduledCircuit
from tqec.compile.blocks.enums import SpatialBlockBorder
from tqec.compile.blocks.layers.atomic.base import BaseLayer
from tqec.utils.scale import LinearFunction, PhysicalQubitScalable2D

Coord2D = tuple[int, int]
"""A qubit coordinate in a layer's local element frame."""


@runtime_checkable
class FlowSpecLayer(Protocol):
    """A raw round that describes its stabilizer flows to the annotators.

    A :class:`RawCircuitLayer` provides a circuit but no template, so the
    detector annotator cannot recover the round's stabilizers the way it does
    for a :class:`~tqec.compile.blocks.layers.atomic.plaquettes.PlaquetteLayer`.
    A raw round that is one round of a longer construction (the rounds of the
    Y-basis measurement cap) instead states its flows explicitly, and the
    annotators build the cross-round detectors and the logical readout from
    them.

    Every spec is keyed by, and made of, **qubit coordinates in the round's
    local element frame** --- never measurement record indices, which do not
    survive the interleaving performed when this round is merged with a
    coexisting round at another position.

    """

    def start_spec(self, k: int) -> Mapping[Coord2D, Sequence[Coord2D]]:
        """Qubits this round measures to detect each stabilizer.

        Keyed by the stabilizer's ancilla coordinate, matched against the
        *previous* round's measurement of that same stabilizer.
        """
        ...

    def end_spec(self, k: int) -> Mapping[Coord2D, Sequence[Coord2D]] | None:
        """Qubits this round measures while *preparing* each stabilizer.

        ``None`` when the next round can recover the match on its own, which it
        can whenever this round measures every stabilizer with a single ancilla
        (every standard round).
        """
        ...

    def observable_spec(self, k: int) -> Sequence[Coord2D] | None:
        """Qubits whose measurements reconstruct a logical operator, if any."""
        ...

    def reconstruction_spec(self, k: int) -> Mapping[Coord2D, Sequence[Coord2D]] | None:
        """Stabilizers reconstructed within this round, if any.

        Keyed by ancilla coordinate; the values are the qubits (ancilla plus
        transversally measured data qubits) whose parity is deterministic.
        """
        ...


class RawCircuitLayer(BaseLayer):
    def __init__(
        self,
        circuit_factory: Callable[[int], ScheduledCircuit],
        scalable_raw_shape: PhysicalQubitScalable2D,
        scalable_num_moments: LinearFunction,
        trimmed_spatial_borders: frozenset[SpatialBlockBorder] = frozenset(),
    ):
        """Represent a layer with a spatial footprint that is defined by a raw circuit.

        Args:
            circuit_factory: a function callable returning a quantum circuit for
                any input ``k >= 1``.
            scalable_raw_shape: scalable shape of the quantum circuit returned
                by the provided ``circuit_factory``.
            scalable_num_moments: a linear function associating to any input
                ``k >= 1`` the number of moments returned by the provided
                ``circuit_factory`` when given ``k`` as input. This is expected
                to be constant in most scenario, but left as a linear function
                to cover potential edge-cases.
            trimmed_spatial_borders: all the spatial borders that have been
                removed from the layer.

        """
        super().__init__(trimmed_spatial_borders)
        self._circuit_factory = circuit_factory
        self._scalable_raw_shape = scalable_raw_shape
        self._scalable_num_moments = scalable_num_moments

    @property
    def scalable_raw_shape(self) -> PhysicalQubitScalable2D:
        """Get the scalable shape of the quantum circuit returned by ``self.circuit_factory``."""
        return self._scalable_raw_shape

    @property
    def circuit_factory(self) -> Callable[[int], ScheduledCircuit]:
        """Get the callable used to generate a scalable quantum circuit from the scaling factor."""
        return self._circuit_factory  # pragma: no cover

    @property
    @override
    def scalable_shape(self) -> PhysicalQubitScalable2D:
        return self.scalable_raw_shape

    @override
    def with_spatial_borders_trimmed(
        self, borders: Iterable[SpatialBlockBorder]
    ) -> RawCircuitLayer:
        raise NotImplementedError(
            f"Cannot trim spatial borders of a {RawCircuitLayer.__name__} instance."
        )

    @override
    def __eq__(self, value: object) -> bool:
        raise NotImplementedError()

    def __hash__(self) -> int:
        raise NotImplementedError(f"Cannot hash efficiently a {type(self).__name__}.")

    @property
    @override
    def scalable_num_moments(self) -> LinearFunction:
        return self._scalable_num_moments  # pragma: no cover
