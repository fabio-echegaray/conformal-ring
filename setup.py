import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name='ring',
    version='0.1',
    scripts=[],
    author="Fabio Echegaray",
    author_email="fabio.echegaray@gmail.com",
    description="A package that performs a conformal mapping from a band around a polygon"
                " into a straight image. "
                "Originally designed for microscope data.",
    # long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fabio-echegaray/ring",
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas>=1.0',
        'matplotlib>=3.0',
        'Shapely>=1.6',
        'pyvista>=0.25.3'
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: AGPL-3.0",
        "Operating System :: OS Independent",
        ],
    )
