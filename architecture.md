# Architecture
## Build System
`pgop` uses CMake for building its C++ code, and `scikit-build-core` as a build back end (for installation via Python tools like `pip` and `build`).
We use install's `DIRECTORY` and `FILES_MATCHING PATTERN` features to automatically install all python files in the `pgop` directory.
Any non-Python files would need to be installed in a separate command.

## Platform Support
Currently `pgop` only supports Unix-like platforms, but nothing in the code should prevent extending to Windows.

## Python Code
The Python is located in the `pgop` directory.
Most of the files are for user use; however, `pgop/generate` provides files to generate data files shipped with PGOP.
The current data files are
* data.h5 - Stores Wigner D matrices for symmetrizing bond order diagrams.
* sphere-codes.npz - Stores the optimal locations of points on a sphere for the Tammes problem.
  This is used for the ``Mesh`` optimizer's ``with_grid`` method.

## Native Extensions
We use C++, located in `src/` to perform the computational complex parts of PGOP.
Furthermore, we use `pybind11 <https://pybind11.readthedocs.io/en/stable/>__` to link our C++ code to the CPython interpreter.

## C++ Structure
The code is broken into a few small sections
- PGOP: the heart of the algorithm and the primary interface of C++ to Python
- optimize: Various optimizers used for finding optimal rotations for PGOP.
- data: Data types used to facilitate the computation of PGOP
- util: Sundry classes and methods used to aid/simplify computation of PGOP

## CI
The code uses pre-commit to format and lint code before committing.

## Testing
Testing is done through `pytest`.

## Adding New Point Groups
To add a new point group, `pgop/data.h5` needs to be regenerated with the new Wigner D matrix (there is one exception).
We store all the point group Wigner D matrices except those that are formed by a semi-direct product with a reflection or inversion.
The semi-direct product composes the actions of both groups leading to a multiplicative addition to symmetry operations.
Currently, there are no supported point groups that require a reflection semi-direct product, so none are stored in `pgop/data.h5` but `C_i` or inversion is.
The procedure to add a new point group is thus,
1. Add needed rotation quaternions (if any) to `pgop/generate/rotations.py`.
2. Add point group to `pgop/generate/generate.py`.
3. Run `pgop/generate/generate.py` (the file will be saved in the correct location).
4. Add logic to `pgop.wignerd._WignerData` and `pgop.wignerd.WignerD` to correctly get the WignerD matrix.
   a. If the semi-direct product is needed it should be computed here.
   b. You may need to add logic to `pgop.wignerd._parse_point_group` to correct get the point group information.
5. Test the new point group by testing a known shape with the given symmetry as the local environment.
