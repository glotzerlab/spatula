=======
SPATULA
=======

Overview
--------

``SPATULA`` (Symmetry Pattern Analysis Toolkit for Understanding Local Arrangements) is a Python package for computing the continuous symmetry ordering of the neighbors of a point in space.
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
- `Installation Guide <https://spatula.readthedocs.io/en/latest/installation.html>`__: Instructions for installing and compiling **spatula**.
- `GitHub repository <https://github.com/glotzerlab/spatula>`__: Download the **spatula** source code.
- `Issue tracker <https://github.com/glotzerlab/spatula/issues>`__: Report issues or request features.

Related Tools
=============

- `HOOMD-blue <https://hoomd-blue.readthedocs.io/>`__: Perform MD / MC simulations that
  can be analyzed with **spatula**.
- `freud <https://freud.readthedocs.io/>`__: Analyze particle simulations.
- `signac <https://signac.readthedocs.io/>`__: Manage your workflow with **signac**.

Citation
========

When using **spatula** to process data for publication, please refer to the `documentation instructions
<https://spatula.readthedocs.io/en/latest/citing.html>`__.


Installation
============
**Spatula** is available on PyPI and conda-forge.
See the Installation Guide for more information.

Example
-------

.. code-block:: python

    import freud
    import spatula

    system = freud.data.UnitCell.fcc().generate_system(3)
    optimizer = spatula.optimize.Union.with_step_gradient_descent(
        optimizer=spatula.optimize.Mesh.from_grid()
    )
    PGOP_Oh_Ih = spatula.PGOP(["Oh","Ih"], optimizer)
    PGOP_Oh_Ih.compute(system, sigmas=None, neighbors={"r_max": 1.2, "exclude_ii": True})
    print(PGOP_Oh_Ih.order)
