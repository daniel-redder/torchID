from setuptools import setup

from torchID import __version__

setup(
    name="torchID",
    version=__version__,
    author="Daniel Redder",
    author_email="daniel@redder.dev",
    description="A package for identifying tensors in PyTorch",
    py_modules=["torchID"],
)