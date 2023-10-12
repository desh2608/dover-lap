import os

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

long_description = open("README.md").read()

dev_requires = ["pre-commit", "black", "flake8"]

setup(
    name="dover-lap",
    version="1.3.2",
    author="Desh Raj",
    author_email="r.desh26@gmail.com",
    description="Combine overlap-aware diarization output RTTMs",
    keywords="diarization dover",
    url="https://github.com/desh2608/dover-lap",
    license="Apache-2.0 License",
    packages=find_packages(),
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    entry_points={"console_scripts": ["dover-lap=dover_lap.dover_lap:main"]},
)
