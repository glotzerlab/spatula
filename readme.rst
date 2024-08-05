====
PGOP
====

Overview
--------

``pgop`` is a Python package for computing the fractional point group ordering of the neighbors of a point in space.
In general, this is to compute the local ordering of particles (molecules) in simulations or experiments over time.
The package serves as an extension of `freud <https://github.com/glotzerlab/freud>`__ with a new order parameter.

``pgop`` currently supports all point groups of finite order.

Installation
------------
To install clone the repository and install using pip.
Install requires a modern (>=3.15) version of CMake.

.. code-block:: bash

   git clone https://github.com/glotzerlab/pgop.git --depth=1
   cd pgop
   python3 -m pip install .


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
