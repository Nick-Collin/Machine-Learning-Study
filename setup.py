from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neurolite",
    version="0.1.0",
    description="A simple neural network framework for educational purposes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nicolas Pinho",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
