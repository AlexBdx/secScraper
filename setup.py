import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="secScraper",
    version="0.0.30",
    author="Alex Bondoux",
    author_email="alexandre.bdx@gmail.com",
    description="Library for Insight project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlexBdx/sec_scrapper/tree/master/sec_scrapper",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
    ],
)
