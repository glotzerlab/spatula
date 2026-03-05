.. highlight:: shell

============
Installation
============

**spatula** binaries are available on conda-forge_ and PyPI_. You can also compile **spatula** from
source.

Binaries
--------

conda-forge package
^^^^^^^^^^^^^^^^^^^

**spatula** is available on conda-forge_ for the *linux-64*, *osx-64*, *osx-arm64* and *win-64*
architectures. Execute one of the following commands to install **spatula**:

.. tab:: Pixi

   .. code-block:: bash

      pixi add spatula

.. tab:: Micromamba

   .. code-block:: bash

      micromamba install spatula

.. tab:: Mamba

   .. code-block:: bash

      mamba install spatula

PyPI
^^^^

Use **uv** or **pip** to install **spatula** binaries from PyPI_ into a virtual environment:

.. tab:: uv

   .. code-block:: bash

      uv pip install spatula-analysis

.. tab:: pip

   .. code-block:: bash

      python3 -m pip install spatula-analysis

.. _conda-forge: https://conda-forge.org/
.. _PyPI: https://pypi.org/
.. _ISPC: https://ispc.github.io


Compile from source
-------------------

The following are **required** for building and installing **spatula** from source:

- A C++17-compliant compiler
- `Python <https://www.python.org/>`__
- `NumPy <https://www.numpy.org/>`__
- `Scipy <https://scipy.org/>`__
- `freud <https://freud.readthedocs.io/en/latest/>`__
- `nanobind <https://nanobind.readthedocs.io/en/latest/>`__
- `scikit-build-core <https://scikit-build-core.readthedocs.io/en/latest/index.html>`__
- `CMake <https://cmake.org/>`__

.. code-block:: bash

    mamba install -c conda-forge cxx-compiler numpy scipy freud nanobind scikit-build-core cmake

All requirements other than the compiler can also be installed via the `Python Package Index <https://pypi.org/>`__

.. code-block:: bash

    uv pip install numpy scipy freud-analysis nanobind scikit-build-core cmake

The commands below clone and build **spatula**:

.. code-block:: bash

   git clone https://github.com/glotzerlab/spatula.git
   cd spatula
   python -m pip install .

For faster incremental rebuilds during development, run the following command
after source changes to reuse the existing build directory:

.. code-block:: bash

   uv pip install --no-deps --no-build-isolation --force-reinstall -C build-dir=$PWD/build .

When developing code that makes use of the project's ISPC_
extensions, or when building from source to maximize performance, users must also
install the requisite compiler. This is available from most package managers, or via
conda-forge:


.. tab:: Mamba

   .. code-block:: bash

      mamba install -c conda-forge ispc

.. tab:: Homebrew (macOS)

   .. code-block:: bash

      brew install ispc

.. tab:: apt (Ubuntu)

   .. code-block:: bash

      sudo apt-get install ispc


Building Documentation
----------------------

The documentation can also be built locally.
The required packages are

+ furo
+ sphinx
+ sphinxcontrib-bibtex
+ sphinxcontrib-katex
+ ipython
+ nbsphinx
+ sphinx-inline-tabs

These can be installed with ``python -m pip install sphinx furo sphinxcontrib-bibtex sphinxcontrib-katex ipython nbsphinx sphinx-inline-tabs``.
Navigate to docs folder ``cd docs``.
To build documentation in ``html`` form run ``make html``.
To view the built documentation open the ``index.html`` file in ``./docs/build`` with your preferred browser.
