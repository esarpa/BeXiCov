from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="BeXiCov",
    version="1.0",
    author="Elena Sarpa",
    author_email="sarpa@cppm.in2p3.fr",
    description="routine to generate the best-fit Gaussian covariance matrix representing the mesured 2PCF multipoles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/esarpa/BeXiCov",
    package_dir={"": "."},
    packages=['BeXiCov'],
    python_requires=">=3.6",
    install_requires=[
        "matplotlib", "numpy", "iMinuit","camb","hankl", "scipy" ]
)
