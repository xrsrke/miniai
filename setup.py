"""Setups the project."""
from setuptools import find_packages, setup


def read_file(fpath):
    """Read file."""
    with open(fpath) as fp:
        data = fp.read()
    return data


setup(
    name="miniai",
    version="0.0.1",
    author="xariusrke",
    author_email="b3f0cus@icloud.com",
    maintainer="xariusrke",
    url="https://github.com/xrsrke/miniai",
    python_requires=">=3.7",
    install_requires=read_file("requirements.txt").split("\n"),
    description="A deep learning framework",
    license="GPL-3.0",
    keywords="deep learning, ai, machine learning",
    packages=find_packages(),
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
)
