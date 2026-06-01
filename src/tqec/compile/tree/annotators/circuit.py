from typing_extensions import override

from tqec.compile.blocks.layers.atomic.layout import LayoutLayer
from tqec.compile.tree.node import LayerNode, NodeWalker


class AnnotateCircuitOnLayerNode(NodeWalker):
    def __init__(
        self,
        k: int,
        reschedule_measurements: bool = True,
        condition_rec: int | None = None,
    ):
        """Walker annotating ``stim.Circuit`` instances implementing the node on each leaf node.

        Args:
            k: scaling factor.
            reschedule_measurements: whether to reschedule measurements in the generated circuits
                to be in the same moment. Since each plaquette may have its own measurement
                schedule, setting this may be necessary for hardware that requires
                measurements to be synchronous.
            condition_rec: ``stim`` record offset selecting the conditional-cube branch.
                When supplied, leaf nodes whose ``LayoutLayer`` has a non-empty
                ``conditional_layers`` additionally receive a
                :class:`~tqec.compile.conditional.circuit.ConditionalCircuit` annotation
                via :meth:`LayoutLayer.to_conditional_circuit`. The branch-zero
                :class:`ScheduledCircuit` annotation is always populated as before;
                downstream detector annotation continues to read from it.

        """
        self._k = k
        self._reschedule_measurements = reschedule_measurements
        self._condition_rec = condition_rec

    @override
    def visit_node(self, node: LayerNode) -> None:
        if not node.is_leaf:
            return
        assert isinstance(node._layer, LayoutLayer)
        node.set_circuit_annotation(
            self._k,
            node._layer.to_circuit(self._k, reschedule_measurements=self._reschedule_measurements),
        )
        if self._condition_rec is not None and node._layer.conditional_layers:
            node.set_conditional_circuit_annotation(
                self._k,
                node._layer.to_conditional_circuit(
                    self._k,
                    condition_rec=self._condition_rec,
                    reschedule_measurements=self._reschedule_measurements,
                ),
            )
