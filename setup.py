from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


setup(
    name="torchfaiss",
    version="0.1.0",
    description="Pure PyTorch distributed KMeans with a FAISS-compatible API",
    long_description=README,
    long_description_content_type="text/markdown",
    author="anxiangsir",
    author_email="anxiangsir@outlook.com",
    url="https://github.com/anxiangsir/torchfaiss",
    packages=find_packages(include=["torchfaiss", "torchfaiss.*"]),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "torch>=2.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
