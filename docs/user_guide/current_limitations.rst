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

``Y``-basis measurements
------------------------

Walking codes
-------------

Transversal Hadamard gates for surface code
-------------------------------------------

Conditional cubes
-----------------

A conditional cube (:class:`~tqec.computation.cube.ConditionalLeafCubeKind`) picks
one of two cube kinds at runtime, from the parity of a correlation surface measured
earlier. Its compiled form is Stim text with ``IF``/``ELSE`` blocks, produced by
:meth:`~tqec.compile.graph.TopologicalComputationGraph.generate_conditional_stim_text`;
``stim`` itself cannot simulate it. The following restrictions apply:

* only the four temporal-basis pairs (``ZXZ_ZXX``, ``XZZ_XZX``, ``ZXX_ZXZ``,
  ``XZX_XZZ``) exist; a pair that changes the spatial boundaries does not;
* at most one conditional cube per ``z``-layer;
* a conditional cube's branches may not differ inside a repeated round;
* the condition must read at least one measurement. A surface that only crosses
  data qubits a temporal pipe carries on is rejected;
* a :class:`~tqec.computation.correlation.ConditionalCorrelationSurface` must be
  XOR-decomposable across its conditions, and is only emitted by
  ``generate_conditional_stim_text``: ``generate_stim_circuit`` leaves it out, with a
  warning. The validity of each branch's surface is not checked.
