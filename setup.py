from setuptools import find_packages, setup

import os
from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name="soccertrack",
    packages=find_packages(),
    version="0.1.0",
    description="A short description of the project.",
    author="Atom Scott",
    license="MIT",
    install_requires=required,
)
