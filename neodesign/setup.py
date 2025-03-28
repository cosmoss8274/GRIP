from setuptools import setup, find_packages

setup(
    name="neodesign",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.24.0",
        "biopython>=1.78",
        "einops>=0.3.0",
        "tqdm>=4.50.0",
    ],
    author="Yi Chai",
    author_email="ychai264@gmail.com",
    description="De novo peptide neoantigen design/scoring for MHC and TCR")
