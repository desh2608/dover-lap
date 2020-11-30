import os
from pathlib import Path

from setuptools import find_packages, setup

project_root = Path(__file__).parent

install_requires = (project_root / 'requirements.txt').read_text().splitlines()

setup(
    name = "dover-lap",
    version = "0.1.0",
    author = "Desh Raj",
    author_email = "r.desh26@gmail.com",
    description = "Combine overlap-aware diarization output RTTMs",
    keywords = "diarization dover",
    url = "https://github.com/desh2608/dover-lap",
    license='Apache-2.0 License',
    packages=find_packages(),
    install_requires=install_requires,
    long_description=(project_root / 'README.md').read_text(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    entry_points={
        'console_scripts': [
            'dover-lap=dover_lap.run:main'
        ]
    }
)