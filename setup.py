from setuptools import setup, find_packages

setup(
    name="iinfft",
    version="0.1.0",
    description="Interpolative Fourier analysis for irregularly-spaced data",
    author="Michael Sorochan Armstrong",
    author_email="mdarmstr@ugr.es",
    url="https://github.com/mdarmstr/iinfft",  # replace
    packages=find_packages(),
    package_data={
        "iinfft.data": ["*.mat", "*.npy", "*.csv"],
    },
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "h5py",
        "pandas",
        # Default NUFFT backend (CPU)
        "finufft",
    ],
    extras_require={
        # Optional GPU backend
        # Usage: pip install .[gpu]
        "gpu": [
            "cufinufft","torch"
        ],
        # Optional: if you want a "no-nufft" minimal install
        # Usage: pip install .[nonufft]
        "nonufft": [],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
