from setuptools import setup, find_packages

setup(
    name="myvllm",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires="==3.11.14",
    install_requires=[
        "transformers",
        "xxhash",
    ],
    extras_require={
        "xpu": [
            "torch==2.11.0+xpu",
            "torchvision==0.26.0+xpu",
            "torchaudio==2.11.0+xpu",
        ],
    },
)
