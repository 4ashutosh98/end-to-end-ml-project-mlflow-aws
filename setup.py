"""
The setup.py file is an essential part of packaging and
distributing Python projects. It is used by setuptools
(or distutils in older Python versions) to define the configuration
of the project, including the package name, version, 
metadata, dependencies and more
"""

from setuptools import setup, find_packages
from typing import List

def get_requirements() -> List[str]:
    """
    This function reads the requirements from the requirements.txt file
    and returns them.
    """

    requirement_list :List[str] = []
    try:
        with open('requirements.txt', 'r') as f:
            # Read all lines in the file
            lines  = f.readlines()
            ## process each line
            for line in lines:
                requirement = line.strip()
                ## ingoring empty lines and -e.
                if requirement and not requirement == '-e .':
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print("No requirements.txt file found")

    return requirement_list

setup(
    name = "NetworkSecurity",
    version = "0.0.1",
    author = "Ashutosh Choudhari",
    author_email= "4ashutosh98@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements()
)