# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from glob import glob
from os.path import basename, dirname, join, splitext

from setuptools import find_packages, setup

setup(
    name="satclip",
    version="2025.3.1",
    # install_requires=REQUIREMENTS,
    py_modules=[splitext(basename(path))[0] for path in glob('satclip/*.py')],
    include_package_data=True,
    packages=find_packages(),
    package_dir={
        "satclip": "satclip",
    },
    url="https://github.com/microsoft/satclip",
)
