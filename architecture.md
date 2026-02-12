# Architecture
## Build System
`spatula` uses CMake for building its C++ code, and `scikit-build-core` as a build back end (for installation via Python tools like `pip` and `build`).
We use install's `DIRECTORY` and `FILES_MATCHING PATTERN` features to automatically install all python files in the `spatula` directory.
Any non-Python files would need to be installed in a separate command.

## Platform Support
Currently `spatula` only supports Unix-like platforms, but nothing in the code should prevent extending to Windows.

## Python Code
The Python is located in the `spatula` directory.
Most of the files are for user use.
The current data files are
* sphere-codes.npz - Stores the optimal locations of points on a sphere for the Tammes problem.
  This is used for the ``Mesh`` optimizer's ``with_grid`` method.

## Native Extensions
We use C++, located in `src/` to perform the computational complex parts of spatula.
Furthermore, we use `nanobind <https://nanobind.readthedocs.io/en/latest/>__` to link our C++ code to the CPython interpreter.

## C++ Structure
The code is broken into a few small sections
- spatula: the heart of the algorithm and the primary interface of C++ to Python
- optimize: Various optimizers used for finding optimal rotations for spatula.
- data: Data types used to facilitate the computation of spatula
- util: Sundry classes and methods used to aid/simplify computation of spatula

## CI
The code uses prek to format and lint code before committing.

## Testing
Testing is done through `pytest`.

## Documentation
`sphinx-doc` NumPy styling is used for Python code.
Doxygen documentation styling is used for C++ code.
