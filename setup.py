from setuptools import setup, find_packages

setup(
    name="centered_set_inference",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.18.0",
        "numpy>=1.20.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "gradio>=2.9.0",
        "pandas>=1.3.0",
        "sentence-transformers>=2.2.0",
        "psutil>=5.9.0"
    ],
    author="Gregory M. Wiedeman",
    author_email="gwiedeman@live.com",
    description="A proof-of-concept implementation of the Centered Set Inference framework for AI alignment",
    keywords="AI, alignment, centered set, inference, language models",
    python_requires=">=3.7",
) 