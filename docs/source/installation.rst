.. highlight:: shell

============
Installation
============

Compile from source
-------------------

The following are **required** for building and installing **spatula** from source:

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

    uv pip install numpy scipy freud-analysis pybind11 scikit-build-core cmake

The code that follows builds **spatula**:

.. code-block:: bash

    git clone https://github.com/glotzerlab/spatula.git
    cd spatula
    python -m pip install .


Building Documentation
----------------------

The documentation can also be built locally.
The required packages are

+ furo
+ sphinx
+ sphinxcontrib-bibtex

These can be installed with ``python -m pip install sphinx furo``.
Navigate to docs folder ``cd docs``.
To build documentation in ``html`` form run ``make html``.
To view the built documentation open the ``index.html`` file in ``./docs/build`` with your preferred browser.
