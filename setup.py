# -*- coding: utf-8 -*-
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import io
import os
import pathlib

import setuptools  # type: ignore

package_root = pathlib.Path(__file__).parent.resolve()

name = "google-generativeai"

description = "Google Generative AI High level API client library and tools."


def get_version():
    version = {}
    version_source = (package_root / "google/generativeai/version.py").read_text()
    exec(version_source, version)
    version = version["__version__"]
    return version


version = get_version()

if version[0] == "0":
    release_status = "Development Status :: 4 - Beta"
else:
    release_status = "Development Status :: 5 - Production/Stable"

dependencies = [
    "google-ai-generativelanguage==0.4.0",
    "google-auth>=2.15.0",  # 2.15 adds API key auth support
    "google-api-core",
    "protobuf",
    "pydantic",
    "tqdm",
    "typing-extensions",
]

extras_require = {
    "dev": ["absl-py", "black", "nose2", "pandas", "pytype", "pyyaml", "Pillow", "ipython"],
}

url = "https://github.com/google/generative-ai-python"

readme = (package_root / "README.md").read_text()

packages = [
    package for package in setuptools.PEP420PackageFinder.find() if package.startswith("google")
]

namespaces = ["google"]

setuptools.setup(
    name=name,
    version=version,
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Google LLC",
    author_email="googleapis-packages@google.com",
    license="Apache 2.0",
    url=url,
    classifiers=[
        release_status,
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",  # Colab
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    platforms="Posix; MacOS X; Windows",
    packages=packages,
    python_requires=">=3.9",
    namespace_packages=namespaces,
    install_requires=dependencies,
    extras_require=extras_require,
    include_package_data=True,
    zip_safe=False,
)
