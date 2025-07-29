#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name="cpl",
    version=1.0,
    author="lxy",
    url="https://github.com/yefeng23323/CPL",
    description="Code for 'Conditional Prototype Learning for Few-Shot Object Detection(CPL).'",
    packages=find_packages(exclude=('configs', 'data', 'work_dirs')),
    install_requires=['clip@git+ssh://git@github.com/openai/CLIP.git'],
)


