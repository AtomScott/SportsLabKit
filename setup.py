
from setuptools import find_packages, setup


with open("requirements.txt") as f:
    required = f.read().splitlines()


setup(
    name="sportslabkit",
    packages=find_packages(),
    version="0.1.0",
    description="A short description of the project.",
    author="Atom Scott",
    license="MIT",
    install_requires=required,
    entry_points={
        "console_scripts": [
            "soccertrack = soccertrack.cli:main",
        ]
    },
)
