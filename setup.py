from pathlib import Path

from setuptools import find_packages, setup

with open(Path(__file__).resolve().parent / "README.md") as f:
    readme = f.read()

setup(
    name="timethin",
    url="https://github.com/clbarnes/timethin",
    author="Chris L. Barnes",
    description="CLI for filtering strings with datetimes in them",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["timethin"]),
    install_requires=[],
    python_requires=">=3.7, <4.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={"console_scripts": ["timethin=timethin:main"]},
)
