=======
SPATULA
=======

Overview
--------

``SPATULA`` (Symmetry Pattern Analysis Toolkit for Understanding Local Arrangements) is a Python package for computing the continuous point group ordering of the neighbors of a point in space.
In general, this is to compute the local ordering of particles (molecules) in simulations or experiments over time.
The package serves as an extension of `freud <https://github.com/glotzerlab/freud>`__ with a new order parameter.

``spatula`` currently supports all point groups of finite order:

- All crystallographic point groups
- Cyclical groups :math:`C_n`
- Cyclical groups with vertical reflection :math:`C_{nv}`
- Cyclical groups with horizontal reflection :math:`C_{nh}`
- Dihedral groups :math:`D_n`
- Dihedral groups with horizontal reflection :math:`D_{nh}`
- Dihedral groups with diagonal reflections :math:`D_{nd}`
- Polyhedral groups :math:`T, T_h, T_d, O, O_h, I, I_h`
- Rotoreflection groups :math:`S_n`
- Inversion group: :math:`C_i`
- Reflection group: :math:`C_s`

Resources
=========

- `Reference Documentation <https://spatula.readthedocs.io/>`__: Examples, tutorials, and package Python APIs.
- `Installation Guide <https://spatula.readthedocs.io/en/stable/gettingstarted/installation.html>`__: Instructions for installing and compiling **spatula**.
- `GitHub repository <https://github.com/glotzerlab/spatula>`__: Download the **spatula** source code.
- `Issue tracker <https://github.com/glotzerlab/spatula/issues>`__: Report issues or request features.

Related Tools
=============

- `HOOMD-blue <https://hoomd-blue.readthedocs.io/>`__: Perform MD / MC simulations that
  can be analyzed with **spatula**.
- `freud <https://freud.readthedocs.io/>`__: Analyze particle simulations.
- `signac <https://signac.io/>`__: Manage your workflow with **signac**.

Citation
========

When using **spatula** to process data for publication, please `cite the github repository
<https://github.com/glotzerlab/spatula>`__.


Installation
============
Currently, **spatula** is only available as a source package.
See the Installation Guide to compile **spatula** from source.

Example
-------

.. code-block:: python

    import freud
    import spatula

    system = freud.data.UnitCell.fcc().generate_system(5)
    optimizer = spatula.optimize.Union.with_step_gradient_descent(
        optimizer=spatula.optimize.Mesh.from_grid()
    )
    compute = spatula.PGOP(symmetries, optimizer)
    compute.compute(system, {"num_neighbors": 12, "exclude_ii": True})
    print(compute.order)
