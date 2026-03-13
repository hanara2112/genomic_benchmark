from setuptools import setup, find_packages

setup(
    name="dc-genomics-review",
    version="0.1.0",
    description="Standalone review repo for DeepChem genomic sequence support",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "deepchem",
        "numpy",
    ],
    extras_require={
        "torch": [
            "torch",
            "transformers",
        ],
        "data": [
            "genomic-benchmarks",
        ],
        "all": [
            "torch",
            "transformers",
            "genomic-benchmarks",
            "safetensors",
        ],
    },
)
