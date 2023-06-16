import glob
import pathlib
import shutil
import subprocess
import typing

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CMakeBuildHook(BuildHookInterface):
    PLUGIN_NAME = "hatch-cmake"

    def initialize(
        self, version: str, build_data: typing.Dict[str, typing.Any]
    ) -> None:
        cmake_cmd = "cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release"
        print("Running CMake.")
        with pathlib.Path("cmake.log")("w").open as fh:
            subprocess.run(cmake_cmd, shell=True, stdout=fh, stderr=fh)
        build_cmd = "cmake --build build"
        print("Building shared libraries.")
        with pathlib.Path("cmake-build.log").open("w") as fh:
            subprocess.run(build_cmd, shell=True, stdout=fh, stderr=fh)
        shared_libraries = glob.glob("build/src/*.so")
        print("Copying files to package directory.")
        for filename in shared_libraries:
            new_filename = str(
                pathlib.Path("pgop") / pathlib.Path(filename).name
            )
            shutil.copyfile(filename, new_filename)
            build_data["artifacts"].append(new_filename)

    def clean(self, versions: list[str]) -> None:
        shutil.rmtree("build")
