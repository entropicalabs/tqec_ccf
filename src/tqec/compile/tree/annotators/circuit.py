from typing_extensions import override

from tqec.compile.blocks.layers.atomic.layout import LayoutLayer
from tqec.compile.tree.node import LayerNode, NodeWalker


class AnnotateCircuitOnLayerNode(NodeWalker):
    def __init__(
        self,
        k: int,
        reschedule_measurements: bool = True,
        condition_recs: dict[int, list[int]] | None = None,
        min_z: int = 0,
    ):
        """Walker annotating ``stim.Circuit`` instances implementing the node on each leaf node.

        Args:
            k: scaling factor.
            reschedule_measurements: whether to reschedule measurements in the generated circuits
                to be in the same moment. Since each plaquette may have its own measurement
                schedule, setting this may be necessary for hardware that requires
                measurements to be synchronous.
            condition_recs: mapping from z-layer index to the ``stim`` record offsets whose XOR
                selects the conditional-cube branch active on that z-layer. When supplied, each
                leaf whose ``LayoutLayer`` has a non-empty ``conditional_layers`` additionally
                receives a :class:`~tqec.compile.conditional.circuit.ConditionalCircuit`
                annotation via :meth:`LayoutLayer.to_conditional_circuit`. The branch-zero
                :class:`ScheduledCircuit` annotation is always populated as before;
                downstream detector annotation continues to read from it.
            min_z: minimum z-coordinate present in the underlying
                :class:`BlockGraph`. The walker uses it to translate its
                tree-depth position into a ``condition_recs`` key.

        """
        self._k = k
        self._reschedule_measurements = reschedule_measurements
        self._condition_recs = condition_recs
        self._min_z = min_z
        self._depth = 0
        self._z_index = -1

    @override
    def enter_node(self, node: LayerNode) -> None:
        self._depth += 1
        if self._depth == 2:
            self._z_index += 1

    @override
    def exit_node(self, node: LayerNode) -> None:
        self._depth -= 1

    @override
    def visit_node(self, node: LayerNode) -> None:
        if not node.is_leaf:
            return
        assert isinstance(node._layer, LayoutLayer)
        node.set_circuit_annotation(
            self._k,
            node._layer.to_circuit(self._k, reschedule_measurements=self._reschedule_measurements),
        )
        if self._condition_recs is not None and node._layer.conditional_layers:
            recs_here = self._condition_recs.get(self._min_z + self._z_index)
            if recs_here is None:
                return
            node.set_conditional_circuit_annotation(
                self._k,
                node._layer.to_conditional_circuit(
                    self._k,
                    condition_recs=recs_here,
                    reschedule_measurements=self._reschedule_measurements,
                ),
            )
