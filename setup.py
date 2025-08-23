"""
TICE Curvature Intelligence Agent
A symbolic analytics engine for AI alignment, validator coherence, and entropy-aware decision systems.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tice-curvature-agent",
    version="1.0.0",
    author="quantumblackswan",
    author_email="quantum@blackswan.dev",
    description="A symbolic analytics engine for AI alignment and entropy-aware decision systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/quantumblackswan/TICE-CURVATURE-AGENT",
    packages=find_packages(),
    py_modules=["tice", "metrics", "fastapi_service"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "full": [
            "torch>=2.0.0",
            "datasets>=2.0.0",
            "streamlit>=1.28.0",
            "huggingface-hub>=0.17.0",
            "matplotlib>=3.5.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black",
            "flake8",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "tice-server=TICE_plug_newest:main",
            "tice-demo=examples.run_tice_sim:main",
        ],
    },
    include_package_data=True,
    keywords="AI alignment, curvature, entropy, symbolic reasoning, AGI",
    project_urls={
        "Bug Reports": "https://github.com/quantumblackswan/TICE-CURVATURE-AGENT/issues",
        "Source": "https://github.com/quantumblackswan/TICE-CURVATURE-AGENT",
        "Documentation": "https://github.com/quantumblackswan/TICE-CURVATURE-AGENT/blob/main/README.md",
    },
)