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

An in-place ``Y``-basis *measurement* --- a :class:`.YHalfCube` capping a
column, i.e. the readout half of an ``S`` gate --- is implemented for the
fixed-bulk convention, following `Gidney's construction
<https://quantum-journal.org/papers/q-2024-04-08-1310/>`_. The cap may coexist
in a ``z``-slice with cubes that keep running, and its logical readout closes a
correlation surface like any other, so the ``S``-gate observable is compiled
without user intervention.

The following are **not** implemented yet:

* ``Y``-basis **initialization** (a ``Y`` cube with a pipe above it rather than
  below). Only the measurement half of the construction is lowered.
* the same cube under the **fixed-boundary** convention, which still raises
  ``NotImplementedError``.
* a ``Y`` cube directly connected to a ``Port``.

Walking codes
-------------

Transversal Hadamard gates for surface code
-------------------------------------------
