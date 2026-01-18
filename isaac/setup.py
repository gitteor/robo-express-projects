from setuptools import setup, find_packages

setup(
    name="robo-express",
    version="0.1.0",
    description="Box stacking RL environment for Doosan E0509 with IsaacLab",
    author="Robo Express Team",
    packages=find_packages(where="exts"),
    package_dir={"": "exts"},
    python_requires=">=3.10",
)
