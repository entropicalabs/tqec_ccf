import collections
import dataclasses
from typing import List, TypeVar, Dict, Type, Optional, AbstractSet, cast, Set

import numpy as np
import sinter
import stim

R_XYZ = 0
R_XZY = 1
R_YXZ = 2
R_YZX = 3
R_ZXY = 4
R_ZYX = 5
PERMUTATIONS = [
    {'X': 'X', 'Y': 'Y', 'Z': 'Z'},
    {'X': 'X', 'Y': 'Z', 'Z': 'Y'},
    {'X': 'Y', 'Y': 'X', 'Z': 'Z'},
    {'X': 'Y', 'Y': 'Z', 'Z': 'X'},
    {'X': 'Z', 'Y': 'X', 'Z': 'Y'},
    {'X': 'Z', 'Y': 'Y', 'Z': 'X'},
]
INVERSE_PERMUTATIONS = [
    {'X': 'X', 'Y': 'Y', 'Z': 'Z'},
    {'X': 'X', 'Y': 'Z', 'Z': 'Y'},
    {'X': 'Y', 'Y': 'X', 'Z': 'Z'},
    {'X': 'Z', 'Y': 'X', 'Z': 'Y'},
    {'X': 'Y', 'Y': 'Z', 'Z': 'X'},
    {'X': 'Z', 'Y': 'Y', 'Z': 'X'},
]
ORIENTATIONS = [
    'I',
    'SQRT_X',
    'S',
    'C_XYZ',
    'C_ZYX',
    'H',
]
ORIENTATION_MULTIPLICATION_TABLE = np.array([
    [0, 1, 2, 3, 4, 5],
    [1, 0, 4, 5, 2, 3],
    [2, 3, 0, 1, 5, 4],
    [3, 2, 5, 4, 0, 1],
    [4, 5, 1, 0, 3, 2],
    [5, 4, 3, 2, 1, 0],
], dtype=np.uint8)


class Layer:
    def copy(self) -> 'Layer':
        raise NotImplementedError()

    def touched(self) -> Set[int]:
        raise NotImplementedError()

    def to_z_basis(self) -> List['Layer']:
        return [self]

    def append_into_stim_circuit(self, out: stim.Circuit) -> None:
        raise NotImplementedError()

    def optimized(self, next_layer: Optional['Layer']) -> List[Optional['Layer']]:
        return [self, next_layer]

    def is_vacuous(self) -> bool:
        return False

    def requires_tick_before(self) -> bool:
        return True

    def implies_eventual_tick_after(self) -> bool:
        return True


@dataclasses.dataclass
class AnnotationLayer(Layer):
    circuit: stim.Circuit = dataclasses.field(default_factory=stim.Circuit)

    def copy(self) -> 'Layer':
        return AnnotationLayer(self.circuit.copy())

    def touched(self) -> Set[int]:
        return set()

    def requires_tick_before(self) -> bool:
        return False

    def implies_eventual_tick_after(self) -> bool:
        return False

    def append_into_stim_circuit(self, out: stim.Circuit) -> None:
        out += self.circuit


@dataclasses.dataclass
class ResetLayer(Layer):
    targets: List[int] = dataclasses.field(default_factory=list)
    bases: List[str] = dataclasses.field(default_factory=list)

    def copy(self) -> 'ResetLayer':
        return ResetLayer(targets=list(self.targets), bases=list(self.bases))

    def touched(self) -> Set[int]:
        return set(self.targets)

    def to_z_basis(self) -> List['Layer']:
        return [
            ResetLayer(targets=list(self.targets), bases=['Z'] * len(self.targets)),
            RotationLayer({q: R_XYZ if b == 'Z' else R_ZYX if b == 'X' else R_XZY for q, b in zip(self.targets, self.bases)}),
        ]

    def append_into_stim_circuit(self, out: stim.Circuit) -> None:
        for t, b in zip(self.targets, self.bases):
            out.append('R' + b, [t])


@dataclasses.dataclass
class MeasureLayer(Layer):
    targets: List[int] = dataclasses.field(default_factory=list)
    bases: List[str] = dataclasses.field(default_factory=list)

    def copy(self) -> 'MeasureLayer':
        return MeasureLayer(targets=list(self.targets), bases=list(self.bases))

    def touched(self) -> Set[int]:
        return set(self.targets)

    def to_z_basis(self) -> List['Layer']:
        rot = RotationLayer({q: R_XYZ if b == 'Z' else R_ZYX if b == 'X' else R_XZY for q, b in zip(self.targets, self.bases)})
        return [
            rot,
            MeasureLayer(targets=list(self.targets), bases=['Z'] * len(self.targets)),
            rot.copy(),
        ]

    def append_into_stim_circuit(self, out: stim.Circuit) -> None:
        for t, b in zip(self.targets, self.bases):
            out.append('M' + b, [t])


@dataclasses.dataclass
class MppLayer(Layer):
    targets: List[List[stim.GateTarget]] = dataclasses.field(default_factory=list)

    def copy(self) -> 'MppLayer':
        return MppLayer(targets=[list(e) for e in self.targets])

    def touched(self) -> Set[int]:
        return set(t.value for mpp in self.targets for t in mpp)

    def to_z_basis(self) -> List['Layer']:
        rot = RotationLayer()
        new_targets: List[List[stim.GateTarget]] = []
        for groups in self.targets:
            new_group: List[stim.GateTarget] = []
            for t in groups:
                new_group.append(stim.target_z(t.value))
                if t.is_x_target:
                    rot.append_rotation(R_ZYX, t.value)
                elif t.is_y_target:
                    rot.append_rotation(R_XZY, t.value)
                elif not t.is_z_target:
                    raise NotImplementedError(f'{t=}')
            new_targets.append(new_group)

        return [
            rot,
            MppLayer(targets=new_targets),
            rot.copy(),
        ]

    def append_into_stim_circuit(self, out: stim.Circuit) -> None:
        flat_targets = []
        for group in self.targets:
            for t in group:
                flat_targets.append(t)
                flat_targets.append(stim.target_combiner())
            flat_targets.pop()
        out.append('MPP', flat_targets)


@dataclasses.dataclass
class InteractLayer(Layer):
    targets1: List[int] = dataclasses.field(default_factory=list)
    targets2: List[int] = dataclasses.field(default_factory=list)
    bases1: List[str] = dataclasses.field(default_factory=list)
    bases2: List[str] = dataclasses.field(default_factory=list)

    def touched(self) -> Set[int]:
        return set(self.targets1 + self.targets2)

    def copy(self) -> 'InteractLayer':
        return InteractLayer(
            targets1=list(self.targets1),
            targets2=list(self.targets2),
            bases1=list(self.bases1),
            bases2=list(self.bases2),
        )

    def _rot_layer(self):
        result = RotationLayer()
        for targets, bases in [(self.targets1, self.bases1), (self.targets2, self.bases2)]:
            for q, b in zip(targets, bases):
                result.rotations[q] = R_XYZ if b == 'Z' else R_ZYX if b == 'X' else R_XZY
        return result

    def to_z_basis(self) -> List['Layer']:
        rot = self._rot_layer()
        return [
            rot,
            InteractLayer(targets1=list(self.targets1),
                          targets2=list(self.targets2),
                          bases1=['Z'] * len(self.targets1),
                          bases2=['Z'] * len(self.targets2)),
            rot.copy(),
        ]

    def append_into_stim_circuit(self, out: stim.Circuit) -> None:
        groups = collections.defaultdict(list)
        for k in range(len(self.targets1)):
            gate = self.bases1[k] + 'C' + self.bases2[k]
            t1 = self.targets1[k]
            t2 = self.targets2[k]
            if gate in ['XCZ', 'YCZ', 'YCX']:
                t1, t2 = t2, t1
                gate = gate[::-1]
            if gate in ['XCX', 'YCY', 'ZCZ']:
                t1, t2 = sorted([t1, t2])
            groups[gate].append((t1, t2))
        for gate in sorted(groups.keys()):
            for pair in sorted(groups[gate]):
                out.append(gate, pair)

    def optimized(self, next_layer: Optional['Layer']) -> List[Optional['Layer']]:
        if isinstance(next_layer, SwapLayer):
            return [InteractSwapLayer(i_layer=self.copy(), swap_layer=next_layer.copy())]
        return [self, next_layer]


def _basis_before_rotation(basis: str, rotation: int) -> str:
    return INVERSE_PERMUTATIONS[rotation][basis]


@dataclasses.dataclass
class FeedbackLayer(Layer):
    controls: List[stim.GateTarget] = dataclasses.field(default_factory=list)
    targets: List[int] = dataclasses.field(default_factory=list)
    bases: List[str] = dataclasses.field(default_factory=list)

    def copy(self) -> 'FeedbackLayer':
        return FeedbackLayer(targets=list(self.targets), controls=list(self.controls), bases=list(self.bases))

    def touched(self) -> Set[int]:
        return set(self.targets)

    def requires_tick_before(self) -> bool:
        return False

    def implies_eventual_tick_after(self) -> bool:
        return False

    def before(self, layer: 'RotationLayer') -> 'FeedbackLayer':
        return FeedbackLayer(
            controls=list(self.controls),
            targets=list(self.targets),
            bases=[_basis_before_rotation(b, layer.rotations.get(t, 0)) for b, t in zip(self.bases, self.targets)],
        )

    def append_into_stim_circuit(self, out: stim.Circuit) -> None:
        for c, t, b in zip(self.controls, self.targets, self.bases):
            out.append('C' + b, [c, t])


@dataclasses.dataclass
class LoopLayer(Layer):
    body: 'LayerCircuit'
    repetitions: int

    def copy(self) -> 'LoopLayer':
        return LoopLayer(body=self.body.copy(), repetitions=self.repetitions)

    def touched(self) -> Set[int]:
        return self.body.touched()

    def to_z_basis(self) -> List['Layer']:
        return [LoopLayer(
            body=self.body.to_z_basis(),
            repetitions=self.repetitions,
        )]

    def optimized(self, next_layer: Optional['Layer']) -> List[Optional['Layer']]:
        optimized = LoopLayer(
            body=self.body.with_merged_layers(),
            repetitions=self.repetitions,
        )
        if len(optimized.body.layers) >= 2:
            first_layer = optimized.body.layers[0]
            last_layer = optimized.body.layers[-1]
            if isinstance(first_layer, RotationLayer) and isinstance(last_layer, RotationLayer):
                optimized.body.layers[-1:] = last_layer.optimized(first_layer)
                optimized.body.layers.pop(0)
                return [first_layer, optimized, *first_layer.inverse().optimized(next_layer)]

        return [optimized, next_layer]

    def implies_eventual_tick_after(self) -> bool:
        return False

    def append_into_stim_circuit(self, out: stim.Circuit) -> None:
        body = self.body.to_stim_circuit()
        body.append('TICK')
        out.append(stim.CircuitRepeatBlock(repeat_count=self.repetitions, body=body))


@dataclasses.dataclass
class RotationLayer(Layer):
    rotations: Dict[int, int] = dataclasses.field(default_factory=dict)

    def touched(self) -> Set[int]:
        return {k for k, v in self.rotations.items() if v}

    def copy(self) -> 'RotationLayer':
        return RotationLayer(dict(self.rotations))

    def inverse(self) -> 'RotationLayer':
        return RotationLayer(rotations={q: R_YZX if r == R_ZXY else R_ZXY if r == R_YZX else r for q, r in self.rotations.items()})

    def append_into_stim_circuit(self, out: stim.Circuit) -> None:
        v = sinter.group_by(self.rotations.items(), key=lambda e: e[1])
        for r, items in sorted(v.items(), key=lambda e: ORIENTATIONS[e[0]]):
            if r:
                out.append(ORIENTATIONS[r], sorted(q for q, _ in items))

    def prepend_rotation(self, rotation_index: int, target: int):
        r1 = self.rotations.setdefault(target, R_XYZ)
        self.rotations[target] = ORIENTATION_MULTIPLICATION_TABLE[r1][rotation_index]

    def append_rotation(self, rotation_index: int, target: int):
        r1 = self.rotations.setdefault(target, R_XYZ)
        self.rotations[target] = ORIENTATION_MULTIPLICATION_TABLE[rotation_index][r1]

    def is_vacuous(self) -> bool:
        return not any(self.rotations.values())

    def optimized(self, next_layer: Optional['Layer']) -> List[Optional['Layer']]:
        if isinstance(next_layer, AnnotationLayer):
            return [next_layer, self]
        if isinstance(next_layer, FeedbackLayer):
            return [next_layer.before(self), self]
        if isinstance(next_layer, ResetLayer):
            trimmed = self.copy()
            for t in next_layer.targets:
                if t in trimmed.rotations:
                    del trimmed.rotations[t]
            if trimmed.rotations:
                return [trimmed, next_layer]
            else:
                return [next_layer]
        if isinstance(next_layer, RotationLayer):
            result = RotationLayer(rotations=dict(self.rotations))
            for q, r in next_layer.rotations.items():
                result.append_rotation(r, q)
            return [result]
        return [self, next_layer]


@dataclasses.dataclass
class SwapLayer(Layer):
    targets1: List[int] = dataclasses.field(default_factory=list)
    targets2: List[int] = dataclasses.field(default_factory=list)

    def touched(self) -> Set[int]:
        return set(self.targets1 + self.targets2)

    def copy(self) -> 'SwapLayer':
        return SwapLayer(targets1=list(self.targets1), targets2=list(self.targets2))

    def append_into_stim_circuit(self, out: stim.Circuit) -> None:
        pairs = []
        for k in range(len(self.targets1)):
            t1 = self.targets1[k]
            t2 = self.targets2[k]
            t1, t2 = sorted([t1, t2])
            pairs.append((t1, t2))
        for pair in sorted(pairs):
            out.append("SWAP", pair)

    def optimized(self, next_layer: Optional['Layer']) -> List[Optional['Layer']]:
        if isinstance(next_layer, InteractLayer):
            i = next_layer.copy()
            i.targets1, i.targets2 = i.targets2, i.targets1
            return [InteractSwapLayer(i_layer=i, swap_layer=self.copy())]
        return [self, next_layer]


@dataclasses.dataclass
class ISwapLayer(Layer):
    targets1: List[int] = dataclasses.field(default_factory=list)
    targets2: List[int] = dataclasses.field(default_factory=list)

    def copy(self) -> 'ISwapLayer':
        return ISwapLayer(targets1=list(self.targets1), targets2=list(self.targets2))

    def touched(self) -> Set[int]:
        return set(self.targets1 + self.targets2)

    def append_into_stim_circuit(self, out: stim.Circuit) -> None:
        pairs = []
        for k in range(len(self.targets1)):
            t1 = self.targets1[k]
            t2 = self.targets2[k]
            t1, t2 = sorted([t1, t2])
            pairs.append((t1, t2))
        for pair in sorted(pairs):
            out.append("ISWAP", pair)

    def optimized(self, next_layer: Optional['Layer']) -> List[Optional['Layer']]:
        return [self, next_layer]


@dataclasses.dataclass
class InteractSwapLayer(Layer):
    i_layer: InteractLayer
    swap_layer: SwapLayer

    def copy(self) -> 'InteractSwapLayer':
        return InteractSwapLayer(i_layer=self.i_layer.copy(), swap_layer=self.swap_layer.copy())

    def touched(self) -> Set[int]:
        return self.i_layer.touched() | self.swap_layer.touched()

    def append_into_stim_circuit(self, out: stim.Circuit) -> None:
        self.i_layer.append_into_stim_circuit(out)
        out.append('TICK')
        self.swap_layer.append_into_stim_circuit(out)

    def to_z_basis(self) -> List['Layer']:
        pairs_1 = {frozenset([a, b]) for a, b in zip(self.i_layer.targets1, self.i_layer.targets2)}
        pairs_2 = {frozenset([a, b]) for a, b in zip(self.swap_layer.targets1, self.swap_layer.targets2)}
        assert pairs_1 == pairs_2
        pre: RotationLayer
        post: RotationLayer
        pre, _, post = self.i_layer.to_z_basis()
        for x, y in zip(self.swap_layer.targets1, self.swap_layer.targets2):
            post.rotations[x], post.rotations[y] = post.rotations[y], post.rotations[x]
            post.prepend_rotation(R_YXZ, x)
            post.prepend_rotation(R_YXZ, y)
        mid = ISwapLayer(targets1=self.swap_layer.targets1, targets2=self.swap_layer.targets2)
        return [pre, mid, post]

    def optimized(self, next_layer: Optional['Layer']) -> List[Optional['Layer']]:
        return [self, next_layer]


@dataclasses.dataclass
class EmptyLayer(Layer):
    def copy(self) -> 'EmptyLayer':
        return EmptyLayer()

    def touched(self) -> Set[int]:
        return set()

    def append_into_stim_circuit(self, out: stim.Circuit) -> None:
        pass

    def optimized(self, next_layer: Optional['Layer']) -> List[Optional['Layer']]:
        return [next_layer]

    def is_vacuous(self) -> bool:
        return True


TLayer = TypeVar('TLayer')


@dataclasses.dataclass
class LayerCircuit:
    layers: List[Layer] = dataclasses.field(default_factory=list)

    def touched(self) -> Set[int]:
        result = set()
        for layer in self.layers:
            result |= layer.touched()
        return result

    def copy(self) -> 'LayerCircuit':
        return LayerCircuit(layers=[e.copy() for e in self.layers])

    def to_z_basis(self) -> 'LayerCircuit':
        result = LayerCircuit()
        for layer in self.layers:
            result.layers.extend(layer.to_z_basis())
        return result

    def _feed(self, kind: Type[TLayer]) -> TLayer:
        if not self.layers:
            self.layers.append(kind())
        elif isinstance(self.layers[-1], EmptyLayer):
            self.layers[-1] = kind()
        elif not isinstance(self.layers[-1], kind):
            self.layers.append(kind())
        return self.layers[-1]

    def _feed_reset(self, basis: str, targets: List[stim.GateTarget]):
        layer = self._feed(ResetLayer)
        for t in targets:
            layer.bases.append(basis)
            layer.targets.append(t.value)

    def _feed_m(self, basis: str, targets: List[stim.GateTarget]):
        layer = self._feed(MeasureLayer)
        for t in targets:
            layer.bases.append(basis)
            layer.targets.append(t.value)

    def _feed_mpp(self, targets: List[stim.GateTarget]):
        layer = self._feed(MppLayer)
        start = 0
        end = 1
        while start < len(targets):
            while end < len(targets) and targets[end].is_combiner:
                end += 2
            layer.targets.append(targets[start:end:2])
            start = end
            end += 1

    def _feed_rotate(self, rotation: int, targets: List[stim.GateTarget]):
        layer = self._feed(RotationLayer)
        for t in targets:
            layer.append_rotation(rotation, t.value)

    def _feed_swap(self, targets: List[stim.GateTarget]):
        layer = self._feed(SwapLayer)
        for k in range(0, len(targets), 2):
            layer.targets1.append(targets[k].value)
            layer.targets2.append(targets[k + 1].value)

    def _feed_iswap(self, targets: List[stim.GateTarget]):
        layer = self._feed(ISwapLayer)
        for k in range(0, len(targets), 2):
            layer.targets1.append(targets[k].value)
            layer.targets2.append(targets[k + 1].value)

    def _feed_c(self, basis1: str, basis2: str, targets: List[stim.GateTarget]):
        is_feedback = any(t.is_sweep_bit_target or t.is_measurement_record_target for t in targets)
        if is_feedback:
            layer = self._feed(FeedbackLayer)
            for k in range(0, len(targets), 2):
                c = targets[k]
                t = targets[k + 1]
                if t.is_sweep_bit_target or t.is_measurement_record_target:
                    c, t = t, c
                    layer.bases.append(basis1)
                else:
                    layer.bases.append(basis2)
                layer.controls.append(c)
                layer.targets.append(t.value)
        else:
            layer = self._feed(InteractLayer)
            for k in range(0, len(targets), 2):
                layer.bases1.append(basis1)
                layer.bases2.append(basis2)
                layer.targets1.append(targets[k].value)
                layer.targets2.append(targets[k + 1].value)

    @staticmethod
    def from_stim_circuit(circuit: stim.Circuit) -> 'LayerCircuit':
        result = LayerCircuit()
        for instruction in circuit:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                result.layers.append(LoopLayer(
                    body=LayerCircuit.from_stim_circuit(instruction.body_copy()),
                    repetitions=instruction.repeat_count))

            elif instruction.name == 'R':
                result._feed_reset('Z', instruction.targets_copy())
            elif instruction.name == 'RX':
                result._feed_reset('X', instruction.targets_copy())
            elif instruction.name == 'RY':
                result._feed_reset('Y', instruction.targets_copy())

            elif instruction.name == 'M':
                result._feed_m('Z', instruction.targets_copy())
            elif instruction.name == 'MX':
                result._feed_m('X', instruction.targets_copy())
            elif instruction.name == 'MY':
                result._feed_m('Y', instruction.targets_copy())

            elif instruction.name == 'MR':
                result._feed_m('Z', instruction.targets_copy())
                result._feed_reset('Z', instruction.targets_copy())
            elif instruction.name == 'MRX':
                result._feed_m('X', instruction.targets_copy())
                result._feed_reset('X', instruction.targets_copy())
            elif instruction.name == 'MRY':
                result._feed_m('Y', instruction.targets_copy())
                result._feed_reset('Y', instruction.targets_copy())

            elif instruction.name == 'XCX':
                result._feed_c('X', 'X', instruction.targets_copy())
            elif instruction.name == 'XCY':
                result._feed_c('X', 'Y', instruction.targets_copy())
            elif instruction.name == 'XCZ':
                result._feed_c('X', 'Z', instruction.targets_copy())
            elif instruction.name == 'YCX':
                result._feed_c('Y', 'X', instruction.targets_copy())
            elif instruction.name == 'YCY':
                result._feed_c('Y', 'Y', instruction.targets_copy())
            elif instruction.name == 'YCZ':
                result._feed_c('Y', 'Z', instruction.targets_copy())
            elif instruction.name == 'CX':
                result._feed_c('Z', 'X', instruction.targets_copy())
            elif instruction.name == 'CY':
                result._feed_c('Z', 'Y', instruction.targets_copy())
            elif instruction.name == 'CZ':
                result._feed_c('Z', 'Z', instruction.targets_copy())

            elif instruction.name in ['H', 'SQRT_Y', 'SQRT_Y_DAG']:
                result._feed_rotate(R_ZYX, instruction.targets_copy())
            elif instruction.name in ['H_XY', 'S', 'S_DAG']:
                result._feed_rotate(R_YXZ, instruction.targets_copy())
            elif instruction.name in ['H_YZ', 'SQRT_X', 'SQRT_X_DAG']:
                result._feed_rotate(R_XZY, instruction.targets_copy())
            elif instruction.name == 'C_XYZ':
                result._feed_rotate(R_YZX, instruction.targets_copy())
            elif instruction.name == 'C_ZYX':
                result._feed_rotate(R_ZXY, instruction.targets_copy())
            elif instruction.name in ['I', 'X', 'Y', 'Z']:
                result._feed_rotate(R_XYZ, instruction.targets_copy())

            elif instruction.name in ['DETECTOR', 'OBSERVABLE_INCLUDE', 'QUBIT_COORDS', 'SHIFT_COORDS']:
                result._feed(AnnotationLayer).circuit.append(instruction)

            elif instruction.name in ['ISWAP', 'ISWAP_DAG']:
                result._feed_iswap(instruction.targets_copy())
            elif instruction.name == 'MPP':
                result._feed_mpp(instruction.targets_copy())
            elif instruction.name == 'SWAP':
                result._feed_swap(instruction.targets_copy())

            elif instruction.name == 'TICK':
                result.layers.append(EmptyLayer())

            else:
                raise NotImplementedError(f'{instruction=}')
        return result

    def __repr__(self) -> str:
        result = ['LayerCircuit(layers=[']
        for layer in self.layers:
            r = repr(layer)
            for line in r.split('\n'):
                result.append('\n    ' + line)
            result.append(',')
        result.append('\n])')
        return ''.join(result)

    def with_merged_layers(self) -> 'LayerCircuit':
        new_layers = []
        def do_layer(layer: Optional[Layer]):
            if new_layers:
                new_layers[-1:] = new_layers[-1].optimized(layer)
            else:
                new_layers.append(layer)
            while new_layers and (new_layers[-1] is None or new_layers[-1].is_vacuous()):
                new_layers.pop()
        for e in self.layers:
            for opt in e.optimized(None):
                do_layer(opt)
        do_layer(None)
        return LayerCircuit(layers=new_layers)

    def _resets_at_layer(self, k: int, *, end_resets: Set[int]) -> Set[int]:
        if k >= len(self.layers):
            return end_resets

        layer = self.layers[k]
        if isinstance(layer, ResetLayer):
            return layer.touched()
        if isinstance(layer, LoopLayer):
            return layer.body._resets_at_layer(0, end_resets=set())
        return set()

    def with_rotations_before_resets_removed(self, loop_boundary_resets: Optional[Set[int]] = None) -> 'LayerCircuit':
        all_touched = self.touched()
        if loop_boundary_resets is None:
            loop_boundary_resets = set()
        sets = [layer.touched() for layer in self.layers]
        sets.append(all_touched)
        resets = [self._resets_at_layer(k, end_resets=all_touched) for k in range(len(self.layers))]
        if loop_boundary_resets is None:
            resets.append(all_touched)
        else:
            resets.append(loop_boundary_resets & (set() if len(resets) == 0 else resets[0]))
        new_layers = [layer.copy() for layer in self.layers]

        for k, layer in enumerate(new_layers):
            if isinstance(layer, LoopLayer):
                layer.body = layer.body.with_rotations_before_resets_removed(loop_boundary_resets=self._resets_at_layer(k + 1, end_resets=all_touched))
            elif isinstance(layer, RotationLayer):
                drops = []
                for q, r in layer.rotations.items():
                    if r:
                        k2 = k + 1
                        while k2 < len(sets):
                            if q in sets[k2]:
                                if q in resets[k2]:
                                    drops.append(q)
                                break
                            k2 += 1
                for q in drops:
                    del layer.rotations[q]

        return LayerCircuit([layer for layer in new_layers if not layer.is_vacuous()])


    def with_squashed_rotations(self) -> 'LayerCircuit':
        sets = [layer.touched() for layer in self.layers]
        def scan(qubit: int, start_layer: int, delta: int) -> Optional[int]:
            while True:
                start_layer += delta
                if start_layer < 0 or start_layer >= len(sets):
                    return None
                if isinstance(new_layers[start_layer], RotationLayer) and not new_layers[start_layer].is_vacuous():
                    return start_layer
                if qubit in sets[start_layer]:
                    return None

        new_layers = [layer.copy() for layer in self.layers]
        cur_layer_index = 0
        while cur_layer_index < len(new_layers):
            layer = new_layers[cur_layer_index]
            if isinstance(layer, RotationLayer):
                rewrites = {}
                for q, r in layer.rotations.items():
                    if not r:
                        continue
                    new_layer_index = scan(q, cur_layer_index, -1)
                    if new_layer_index is None:
                        new_layer_index = scan(q, cur_layer_index, +1)
                    if new_layer_index is not None:
                        rewrites[q] = new_layer_index
                    else:
                        break
                else:
                    for q, r in layer.rotations.items():
                        if not r:
                            continue
                        new_layer_index = rewrites[q]
                        new_layer: RotationLayer = cast(RotationLayer, new_layers[new_layer_index])
                        if new_layer_index > cur_layer_index:
                            new_layer.prepend_rotation(r, q)
                        else:
                            new_layer.append_rotation(r, q)
                        if new_layer.rotations.get(q):
                            sets[new_layer_index].add(q)
                        elif q in sets[new_layer_index]:
                            sets[new_layer_index].remove(q)
                    layer.rotations.clear()
                    sets[cur_layer_index].clear()
            elif isinstance(layer, LoopLayer):
                layer.body = layer.body.with_squashed_rotations()
            cur_layer_index += 1
        return LayerCircuit([layer for layer in new_layers if not layer.is_vacuous()])

    def with_rotations_merged_earlier(self) -> 'LayerCircuit':
        sets = [layer.touched() for layer in self.layers]
        def scan(qubit: int, start_layer: int) -> Optional[int]:
            while True:
                start_layer -= 1
                if start_layer < 0:
                    return None
                l = new_layers[start_layer]
                if isinstance(l, RotationLayer) and qubit in l.rotations:
                    return start_layer
                if qubit in sets[start_layer]:
                    return None

        new_layers = [layer.copy() for layer in self.layers]
        cur_layer_index = 0
        while cur_layer_index < len(new_layers):
            layer = new_layers[cur_layer_index]
            if isinstance(layer, RotationLayer):
                rewrites = {}
                for q, r in layer.rotations.items():
                    if not r:
                        continue
                    v = scan(q, cur_layer_index)
                    if v is not None:
                        rewrites[q] = v
                for q, dst in rewrites.items():
                    new_layer: RotationLayer = cast(RotationLayer, new_layers[dst])
                    new_layer.append_rotation(layer.rotations.pop(q), q)
                    sets[cur_layer_index].remove(q)
                    if new_layer.rotations.get(q):
                        sets[dst].add(q)
                    elif q in sets[dst]:
                        sets[dst].remove(q)
            elif isinstance(layer, LoopLayer):
                layer.body = layer.body.with_rotations_merged_earlier()
            cur_layer_index += 1
        return LayerCircuit([layer for layer in new_layers if not layer.is_vacuous()])

    def with_irrelevant_tail_layers_removed(self) -> 'LayerCircuit':
        irrelevant_layer_types_at_end = (
            ResetLayer,
            InteractLayer,
            FeedbackLayer,
            RotationLayer,
            SwapLayer,
            ISwapLayer,
            InteractSwapLayer,
            EmptyLayer,
        )
        result = list(self.layers)
        while len(result) > 0 and isinstance(result[-1], irrelevant_layer_types_at_end):
            result.pop()
        return LayerCircuit(result)

    def to_stim_circuit(self) -> stim.Circuit:
        circuit = stim.Circuit()
        tick_coming = False
        for layer in self.layers:
            if tick_coming and layer.requires_tick_before():
                circuit.append('TICK')
                tick_coming = False
            layer.append_into_stim_circuit(circuit)
            tick_coming |= layer.implies_eventual_tick_after()
        return circuit


def to_z_basis_interaction_circuit(circuit: stim.Circuit) -> stim.Circuit:
    c = LayerCircuit.from_stim_circuit(circuit)
    c = c.with_merged_layers()
    c = c.to_z_basis()
    c = c.with_merged_layers()
    c = c.with_squashed_rotations()
    c = c.with_rotations_merged_earlier()
    c = c.with_rotations_before_resets_removed()
    c = c.with_irrelevant_tail_layers_removed()
    return c.to_stim_circuit()
