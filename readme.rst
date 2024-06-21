====
PGOP
====

Overview
--------

``pgop`` is a Python package for computing the continuous point group ordering of the neighbors of a point in space.
In general, this is to compute the local ordering of particles (molecules) in simulations or experiments over time.
The package serves as an extension of `freud <https://github.com/glotzerlab/freud>`__ with a new order parameter.

``pgop`` currently supports the following point groups:

- All crystallographic point groups
- Cyclical groups :math:`C_n`
- Cyclical groups with inversion :math:`C_{ni}`
- Cyclical groups with vertical reflection :math:`C_{nv}`
- Cyclical groups with horizontal reflection :math:`C_{nh}`
- Dihedral groups :math:`D_n`
- Dihedral groups with horizontal reflection :math:`D_{nh}`
- Dihedral groups with diagonal reflections :math:`D_{nd}`
- Polyhedral groups :math:`T, T_h, O, O_h, I, I_h`
- Rotoreflection groups :math:`S_n`
- Inversion group: :math:`C_i`

Resources
=========

- `Reference Documentation <https://pgop.readthedocs.io/>`__: Examples, tutorials, and package Python APIs.
- `Installation Guide <https://pgop.readthedocs.io/en/stable/gettingstarted/installation.html>`__: Instructions for installing and compiling **PgOP**.
- `GitHub repository <https://github.com/glotzerlab/pgop>`__: Download the **PgOP** source code.
- `Issue tracker <https://github.com/glotzerlab/pgop/issues>`__: Report issues or request features.

Related Tools
=============

- `HOOMD-blue <https://hoomd-blue.readthedocs.io/>`__: Perform MD / MC simulations that
  can be analyzed with **PgOP**.
- `freud <https://freud.readthedocs.io/>`__: Analyze particle simulations.
- `signac <https://signac.io/>`__: Manage your workflow with **signac**.

Citation
========

When using **PgOP** to process data for publication, please `use this citation - CHANGE
THIS LATER
<https://github.com/glotzerlab/pgop>`__.


Installation
============

**PgOP** is available on conda-forge_. Install with:

.. code:: bash

   mamba install -c conda-forge PgOP

**freud** is also available on PyPI_:

.. code:: bash

   python -m pip install PgOP

.. _conda-forge: https://conda-forge.org/
.. _PyPI: https://pypi.org/

If you need more detailed information or wish to install **PgOP** from source, please refer to the
Installation Guide to compile **PgOP** from source.

Example
-------

.. code-block:: python

    import freud
    import pgop

    system = freud.data.UnitCell.fcc().generate_system(5)
    optimizer = pgop.optimize.Union.with_step_gradient_descent(
        optimizer=pgop.optimize.Mesh.with_grid())
    compute = pgop.PGOP(
        dist="fisher", symmetries=("Oh",), optimizer=optimizer)
    compute.compute(system, {"num_neighbors": 12, "exclude_ii": True})
    # Print the optimizer fractional point group ordering for full
    # octahedral ordering Oh.
    print(compute.pgop)
