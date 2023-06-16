import glob
import os
import platform
import shlex
import shutil
import subprocess
import sys
import typing
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from pydantic import BaseModel, Field

_LIBRARY_EXTENSION = "#lext"


def get_platform() -> str:
    plat = platform.system().lower()
    if plat == "darwin":
        return "macos"
    return plat


def in_cibw() -> bool:
    return "HATCH_CMAKE_CIBW" in os.environ or any(
        "CIBW" in var for var in os.environ
    )


def default_generator():
    plat = get_platform()
    if plat == "windows":
        return "Visual Studio"
    elif plat == "darwin":
        return "Xcode"
    elif shutil.which("ninja") != "":
        return "Ninja"
    return "Unix Makefiles"


def get_python_cmake_opts():
    return {
        "PYTHON_EXECUTABLE": sys.executable,
        "PYTHON_VERSION_STRING": platform.python_version(),
    }


def replace_library_suffix(pattern: str) -> str:
    extension = "so" if get_platform() != "windows" else "dll"
    return pattern.replace(_LIBRARY_EXTENSION, extension)


def get_cibw_cmake_opts() -> typing.Dict[str, str]:
    cmake_options = {}
    plat = get_platform()
    if (
        plat == "macos"
        and "ARCHFLAGS" in os.environ
        and "arm64" in os.environ["ARCHFLAGS"]
    ):
        cmake_options["CMAKE_OSX_ARCHITECTURES"] = "arm64"
    return cmake_options


def get_full_cmake_options(
    config: typing.Dict[str, str]
) -> typing.Dict[str, str]:
    options = get_python_cmake_opts()
    if in_cibw():
        options.update(get_cibw_cmake_opts())
    options.update(config)
    return options


class BuildOptions(BaseModel):
    cmake: typing.List[str] = []
    generator: typing.List[str] = []


class Config(BaseModel):
    build_dir: str = Field("build", alias="build-dir")
    force: bool = False
    options: typing.Dict[str, typing.Any] = {}
    generator: typing.Optional[str] = default_generator()
    build_options: BuildOptions = Field(BuildOptions(), alias="build-options")
    src_mapping: typing.Dict[str, str] = Field(alias="src-mapping")


class CMakeBuildHook(BuildHookInterface):
    PLUGIN_NAME = "hatch-cmake"

    def initialize(
        self, version: str, build_data: typing.Dict[str, typing.Any]
    ) -> None:
        # specify that compiled libraries are packaged with Python.
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
        # Only rerun cmake when asked to.
        if self._config.force or not Path(self._config.build_dir).exists():
            self.run_cmd(self.cmake_cmd, "Running CMake.")
        self.run_cmd(self.build_cmd, "Building shared libraries.")
        self.app.display_info("Copying files to package directory.")
        build_data["artifacts"].extend(self.copy_shared_objects())
        self.app.display_success("CMake completed.")

    def run_cmd(self, cmd: str, msg: str, logfile: typing.Optional[str] = None):
        self.app.display_info(msg)
        if logfile is None:
            subprocess.run(cmd, check=True, shell=True)
            return
        with Path(logfile).open("w") as fh:
            subprocess.run(cmd, shell=True, check=True, stdout=fh, stderr=fh)

    def clean(self, versions: typing.List[str]) -> None:
        build_dir = Path(self._config.build_dir)
        if build_dir.exists():
            shutil.rmtree(build_dir)
        for filename in glob.glob(
            str(Path(self.root, "**", "*.so")), recursive=True
        ):
            Path(filename).unlink()

    @property
    def _config(self):
        """Config for Hatch-CMake.

        Prioritizes options in this order:
            1. CIbuildwheel System specific
            2. CIbuildwheel
            3. System specific
            4. General
        """
        if getattr(self, "__config", None) is None:
            plat = get_platform()
            sys_config = self.config.pop(plat, {})
            config = {**self.config, **sys_config}
            if in_cibw():
                cibw_config = self.config.pop("cibuildwheel", {})
                config.update(cibw_config)
                cibw_sys_config = cibw_config.pop(plat, {})
                config.update(cibw_sys_config)
            self.__config = Config(**config)
        return self.__config

    @property
    def cmake_cmd(self):
        cmd = "cmake -B " + self._config.build_dir
        cmd += " -G " + shlex.quote(self._config.generator)
        full_options = get_full_cmake_options(self._config.options)
        options = " ".join(
            "-D" + f"{opt}={val}" for opt, val in full_options.items()
        )
        return cmd + " " + options

    @property
    def build_cmd(self):
        cmd = "cmake --build " + self._config.build_dir
        cmake_options = " ".join(self._config.build_options.cmake)
        if cmake_options != "":
            cmd += " " + cmake_options
        generator_options = " ".join(self._config.build_options.generator)
        if generator_options != "":
            cmd += " -- " + generator_options
        return cmd

    def copy_shared_objects(self):
        artifacts = []
        for pattern, destination in self._config.src_mapping.items():
            pattern = Path(
                self._config.build_dir,
                "src",
                replace_library_suffix(pattern),
            )
            shared_libraries = glob.glob(str(pattern), recursive=True)
            for filename in shared_libraries:
                new_filename = str(Path(destination, Path(filename).name))
                self.app.display_debug(f"Copying {filename} to {new_filename}")
                shutil.copyfile(filename, new_filename)
                # artifacts cannot handle pathlib.Path objects
                artifacts.append(str(new_filename))
        return artifacts
