.. Copyright (c) 2021-2025 The Regents of the University of Michigan
.. Part of SPATULA, released under the BSD 3-Clause License.

Change Log
==========

`spatula <https://github.com/glotzerlab/spatula>`_ releases follow `semantic versioning
<https://semver.org/>`_.


0.2.0 (xxxx-xx-xx)
^^^^^^^^^^^^^^^^^^

*Changed*

* **Project now uses C++ 20 (primarily for std::span)**. This is great for ergonomics and makes a future Eigen3 port much easier (Eigen::Map is very similar)
* Nanobind exports replace pybind11
* Optimizers are now header-only
* Locality code is now header-only
* Vec3 and Quaternion are now header-only
* BondOrder is now header-only
* Metrics and Utils (excluding QlmEval) are now header only
* ``pgop.py::BOOSOP`` code is now in separate file ``boosop.py``.
* Many ``std::vector<std::vector<...>>`` are now vectors of pointers, allowing for copy- and move- free access to python data. Matrix elements are accessed with ``std::span`` and cast to statically-allocated types for performance.
* ``py::array`` are now replaced with ``std::vector`` or ``type*`` pointers
* Implied rotation matrix type (``std::vector<double>``) is now ``typedef RotationMatrix = std::array<double, 9>``

*Removed*

* PGOPStore
* BOOSOPStore
* Unused python bindings (quaternion, vec3, QLMEval, metrics)

*Added*

* ``m_group_sizes`` class method for PGOP, which stores the size of each group (currently, (group order - 1) * 9). Previous code used vector.size, which requires copies and allocations for both individual elements and entire groups.
* RotationMatrix std::array wrapper for fast and strongly typed vector rotations
* ``-DENABLE_PROFILING`` flag to allow for easy profiling


0.1.1 (2025-10-16)
^^^^^^^^^^^^^^^^^^

*Added:*

* Python 3.14 support.
* Python 3.10 and 3.11 wheels on PyPI.
* Pytest mode to run tests for single example per point group.

*Fixed:*

* Windows failures catching.


0.1.0 (2025-09-20)
^^^^^^^^^^^^^^^^^^

*Added:*

* First PyPI release.

*Changed:*

* Documentation improvements.
* CI improvements.

0.0.1 (2025-09-16)
^^^^^^^^^^^^^^^^^^

*Added:*

* First public release.
