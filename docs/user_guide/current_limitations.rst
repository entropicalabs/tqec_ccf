Current known limitations
=========================

This page lists all the known limitations of the ``tqec`` library.

.. note::

    We aim at making this page as exhaustive as possible, but it will never be 100% exhaustive.
    If you think that this page is missing an important entry, please
    `raise an issue <https://github.com/tqec/tqec/issues/new/choose>`_ to let us know.

.. important::

    If any of the feature discussed below is important to you, please let us know!
    We will try to prioritize depending on community feedback. If you do not want to wait,
    please have a look at the :doc:`/contributor_guide` and open a pull request.

Spatial junctions
-----------------

A spatial junction is any cube that has at least 2 pipes in the spatial (``XY``) plane.
These kind of computation require special handling that is not currently implemented.

``Y``-basis measurement and initialization
------------------------------------------

An in-place ``Y``-basis *measurement* --- a ``LeafCubeKind.Y_HALF_CUBE`` capping a
column, i.e. the readout half of an ``S`` gate --- is implemented for the
fixed-bulk convention, following `Gidney's construction
<https://quantum-journal.org/papers/q-2024-04-08-1310/>`_. The cap may coexist
in a ``z``-slice with cubes that keep running, and its logical readout closes a
correlation surface like any other, so the ``S``-gate observable is compiled
without user intervention.

A ``Y``-basis **initialization** --- a ``Y`` cube with a pipe above it rather
than below --- is lowered as the time reverse of the measurement cap. It is one
round longer than the cap (``k + 4`` rounds rather than ``k + 3``). Next to its
fold, a measurement needs one round that is not in the fixed-bulk interaction
order, and the temporal pipe's junction round provides it. An initialization
needs two: the pipe's junction round, and a *handoff* round the initialization
carries itself, in the cap's interaction order.

Both halves are verified to preserve the circuit distance ``2k + 1`` only for
``k <= 2``. At ``k = 3`` they measure ``2k``: the degenerate patch carries
``k`` boundary rounds, which is too few at ``d = 7``. Gidney's own reference
circuit, with the same number of boundary rounds, falls short in exactly the
same way. Use ``k <= 2`` where the full distance matters.

The following are **not** implemented yet, and raise ``NotImplementedError``:

* the same cube under the **fixed-boundary** convention;
* a ``Y`` cube directly connected to a ``Port``;
* a ``Y`` cube attached through a Hadamard temporal pipe.

Injection cubes
---------------

An injection cube (``LeafCubeKind.INJECTION``) starts a logical column in one of
eight single-qubit states. The encoder is **not fault tolerant**: a single fault
in it corrupts the injected state, so a column started this way has circuit
distance 1. The following restrictions apply:

* fixed-bulk convention only; the fixed-boundary convention raises
  ``NotImplementedError``;
* it must sit directly below a regular cube, joined by one temporal pipe: an
  injection cube on a ``Port``, or directly into a ``Y`` cube, is rejected;
* its observable is non-deterministic by design, even for a stabilizer state
  such as ``"0"``: the correlation surface ends open at the injection cube;
* ``"T"`` and ``"T_DAG"`` are not stim gates, so
  :meth:`~tqec.compile.graph.TopologicalComputationGraph.generate_stim_circuit`
  and :meth:`~tqec.compile.graph.TopologicalComputationGraph.generate_crumble_url`
  (and hence :mod:`tqec.simulation`) refuse them; only
  :meth:`~tqec.compile.graph.TopologicalComputationGraph.generate_stim_text` emits
  them, as text stim cannot parse;
* the state is not stored in ``.dae`` or ``.bgraph`` files: an injection cube read
  back from either has the default state ``"i"``;
* two injection cubes in the same ``z``-slice are untested.

Walking codes
-------------

Transversal Hadamard gates for surface code
-------------------------------------------

Conditional cubes
-----------------

A conditional cube (:class:`~tqec.computation.cube.ConditionalLeafCubeKind`) picks
one of two cube kinds at runtime, from the parity of a correlation surface measured
earlier. Its compiled form is Stim text with ``IF``/``ELSE`` blocks, produced by
:meth:`~tqec.compile.graph.TopologicalComputationGraph.generate_stim_text`;
``stim`` itself cannot simulate it, and
:meth:`~tqec.compile.graph.TopologicalComputationGraph.generate_stim_circuit`
refuses such a computation. To simulate one,
:meth:`~tqec.compile.graph.TopologicalComputationGraph.generate_branch_circuit`
returns the ``stim.Circuit`` of a single branch, noisy if given a noise model;
a logical error rate of the computation is an average over its branches.
``generate_stim_text`` takes a noise model too: each arm of an ``IF``/``ELSE``
block is noised in place, so every branch of the text carries the noise it would
on its own. The following restrictions apply:

* only the four temporal-basis pairs (``ZXZ_ZXX``, ``XZZ_XZX``, ``ZXX_ZXZ``,
  ``XZX_XZZ``) exist; a pair that changes the spatial boundaries does not;
* at most one conditional cube per ``z``-layer;
* a conditional cube's branches may not differ inside a repeated round;
* ``compile_block_graph(..., observables="auto")`` only finds the observables
  that touch no conditional cube, which are deterministic whatever branch each
  cube takes. One that crosses a conditional cube depends on its branch: it is
  left out, with a warning, and must be given explicitly as a
  :class:`~tqec.computation.correlation.ConditionalCorrelationSurface`;
* the condition must read at least one measurement. A surface that only crosses
  data qubits a temporal pipe carries on is rejected;
* a :class:`~tqec.computation.correlation.ConditionalCorrelationSurface` must be
  XOR-decomposable across its conditions. Without a conditional cube, it is only
  emitted by ``generate_stim_text``: ``generate_stim_circuit`` leaves it out, with
  a warning. ``generate_branch_circuit`` refuses one whose condition no
  conditional cube reads. The validity of each branch's surface is not checked.
