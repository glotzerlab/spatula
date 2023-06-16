# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Install pybind11."""

import argparse
import copy
import dataclasses
import logging
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

log = logging.getLogger(__name__)


@dataclasses.dataclass
class CMakePackage:
    name: str
    version: str
    url: str
    cmake_options: list[str]


def find_cmake_package(
    name, version, location_variable=None, ignore_system=False
):
    """Find a package with cmake.

    Return True if the package is found
    """
    if location_variable is None:
        location_variable = name + "_DIR"

    find_package_options = ""
    if ignore_system:
        find_package_options += (
            "NO_SYSTEM_ENVIRONMENT_PATH "
            "NO_CMAKE_PACKAGE_REGISTRY NO_CMAKE_SYSTEM_PATH "
            "NO_CMAKE_SYSTEM_PACKAGE_REGISTRY"
        )

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = pathlib.Path(tmpdirname)

        # write the cmakelists file
        with (tmp_path / "CMakeLists.txt", "w").open("w") as f:
            f.write(
                f"""
project(test)
set(PYBIND11_PYTHON_VERSION 3)
cmake_minimum_required(VERSION 3.9)
find_package({name} {version} CONFIG REQUIRED {find_package_options})
"""
            )

        # add the python prefix to the cmake prefix path
        env = copy.copy(os.environ)
        env["CMAKE_PREFIX_PATH"] = sys.prefix

        tmp_path.mkdir("build")
        cmake_out = subprocess.run(
            ["cmake", tmpdirname],
            cwd=tmp_path / "build",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
            env=env,
            encoding="UTF-8",
        )

        log.debug(cmake_out.stdout.strip())

        # if cmake completed correctly, the package was found
        if cmake_out.returncode == 0:
            location = ""
            with (tmp_path / "build" / "CMakeCache.txt").open("w") as f:
                for line in f.readlines():
                    if line.startswith(location_variable):
                        location = line.strip()

            log.info(f"Found {name}: {location}")
            return True
        else:
            log.debug(cmake_out.stdout.strip())
            return False


def install_cmake_package(url, cmake_options):
    """Install a cmake package."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = pathlib.Path(tmpdirname)

        log.info(f"Fetching {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with (tmp_path / "file.tar.gz", "wb").open("w") as f:
            f.write(urllib.request.urlopen(req).read())

        with tarfile.open(tmp_path / "file.tar.gz") as tar:
            tar.extractall(path=tmp_path)
            root = tar.getnames()[0]
            if "/" in root:
                root = pathlib.Path(root).parent

        # add the python prefix to the cmake prefix path
        env = copy.copy(os.environ)
        env["CMAKE_PREFIX_PATH"] = sys.prefix

        log.info(f"Configuring {root}")
        tmp_path.mkdir("build")
        cmake_out = subprocess.run(
            ["cmake", tmp_path / root, f"-DCMAKE_INSTALL_PREFIX={sys.prefix}"]
            + cmake_options,
            cwd=tmp_path / "build",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
            env=env,
            encoding="UTF-8",
        )

        log.debug(cmake_out.stdout.strip())

        if cmake_out.returncode != 0:
            log.error(
                f"Error configuring {root} (run with -v to see detailed "
                "error messages)"
            )
            raise RuntimeError("Failed to configure package")

        log.info(f"Installing {root}")
        cmake_out = subprocess.run(
            ["cmake", "--build", tmp_path / "build", "--", "install"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
            env=env,
            encoding="UTF-8",
        )

        log.debug(cmake_out.stdout.strip())

        if cmake_out.returncode != 0:
            log.error(
                f"Error installing {root} (run with -v to see detailed "
                "error messages)"
            )
            raise RuntimeError("Failed to install package")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Install header-only libraries needed to build HOOMD-blue."
    )
    parser.add_argument(
        "-q", action="store_true", default=False, help="Suppress info messages."
    )
    parser.add_argument(
        "-v",
        action="store_true",
        default=False,
        help="Show debug messages (overrides -q).",
    )
    parser.add_argument(
        "-y",
        action="store_true",
        default=False,
        help="Skip user input and force installation.",
    )
    parser.add_argument(
        "--ignore-system",
        action="store_true",
        default=False,
        help="Ignore packages installed at the system level.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.q:
        log.setLevel(level=logging.WARNING)
    if args.v:
        log.setLevel(level=logging.DEBUG)

    log.info(f"Searching for packages in: {sys.prefix}")

    packages = [
        CMakePackage(
            name="pybind11",
            version="2.0",
            url="https://github.com/pybind/pybind11/archive/v2.6.0.tar.gz",
            cmake_options=["-DPYBIND11_INSTALL=on", "-DPYBIND11_TEST=off"],
        )
    ]

    missing_packages = []
    for package in packages:
        if not find_cmake_package(
            package.name, package.version, ignore_system=args.ignore_system
        ):
            missing_packages.append(package)

    if len(missing_packages) == 0:
        log.info("Done. Found all packages.")
        sys.exit(0)

    if not args.y:
        package_str = ", ".join([p.name for p in missing_packages])
        print(f"*** About to install {package_str} into {sys.prefix}")
        proceed = input("Proceed (y/n)? ")
        if proceed == "n":
            sys.exit(0)

    log.info(f"Installing packages in: {sys.prefix}")

    for package in missing_packages:
        install_cmake_package(package.url, cmake_options=package.cmake_options)

    log.info("Done.")
