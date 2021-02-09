import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GMMClusteringAlgorithms",
    version="0.0.2",
    author="Colin Weber",
    author_email="colin.weber.27@gmail.com",
    description="A data analysis package for PI-ICR Mass Spectroscopy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pypi.org/project/GMMClusteringAlgorithms",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)