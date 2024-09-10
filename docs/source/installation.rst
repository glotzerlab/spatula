.. highlight:: shell

============
Installation
============

Compile from source
-------------------

The following are **required** for building and installing **PgOP** from source:

- A C++17-compliant compiler
- `Python <https://www.python.org/>`__ 
- `NumPy <https://www.numpy.org/>`__ 
- `Scipy <https://scipy.org/>`__ 
- `freud <https://freud.readthedocs.io/en/latest/>`__ 
- `pybind11 <https://pybind11.readthedocs.io/en/stable/index.html>`__ 
- `scikit-build-core <https://scikit-build-core.readthedocs.io/en/latest/index.html>`__ 
- `CMake <https://cmake.org/>`__ 

.. code-block:: bash

    mamba install conda-forge cxx-compiler numpy scipy freud pybind11 scikit-build-core cmake

All requirements other than the compiler can also be installed via the `Python Package Index <https://pypi.org/>`__

.. code-block:: bash

    uv pip install numpy scipy freud-analysis pybind11 scikit-build cmake

The code that follows builds **PgOP** and installs it for all users (append ``--user`` if you wish to install it to your user site directory):

.. code-block:: bash

    git clone --recurse-submodules https://github.com/glotzerlab/freud.git
    cd freud
    uv pip install .


The sources for PgOP can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/glotzerlab/pgop


Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python -m pip install /path/to/clone


.. _Github repo: https://github.com/glotzerlab/pgop
.. _tarball: https://github.com/glotzerlab/pgop/tarball/main


Building Documentation
----------------------

Currently the documentation is not available online, but can be built locally.
The required packages are

+ furo
+ sphinx

These can be installed with ``python3 -m pip install sphinx furo``.
To build documentation in the project base directory run ``python3 -m sphinx ./docs ./docs/_build``.
To view the built documentation open the ``index.html`` file in ``./docs/_build`` with your preferred browser.
