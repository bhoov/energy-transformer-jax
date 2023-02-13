from setuptools import setup, find_packages

requires = []

setup(
    name="energy-transformer",
    description="Energy formulation for the Transformer Block",
    package_dir={"":"."},
    packages=find_packages("."),
    author="Anonymous Submission",
    include_package_data=True,
    install_requires=requires
)