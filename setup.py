#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# 讀取README檔案作為長描述
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# 讀取requirements.txt
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return requirements

setup(
    name="hierarchical-rag-retrieval",
    author="arthur422tp",  # 請修改為您的名稱
    author_email="arthur422tp@gmail.com",  # 請修改為您的郵箱
    description="AI-Powered Legal Document Retrieval Engine based on Hierarchical Clustering & RAG",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/arthur422tp/hierarchical",  # 請修改為您的GitHub repository URL
    project_urls={
        "Bug Tracker": "https://github.com/arthur422tp/hierarchical/issues",
        "Documentation": "https://github.com/arthur422tp/hierarchical#readme",
        "Source Code": "https://github.com/arthur422tp/hierarchical",
        "arXiv Paper": "https://arxiv.org/abs/2506.13607",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black",
            "flake8",
            "isort",
        ],
        "app": [
            "fastapi==0.104.1",
            "uvicorn==0.24.0",
            "python-multipart==0.0.6",
        ],
    },
    # entry_points={
    #     "console_scripts": [
    #         "hierarchical-rag=hierarchical_rag.cli:main",
    #     ],
    # },
    keywords="rag retrieval hierarchical clustering legal nlp ai machine-learning",
    zip_safe=False,
    include_package_data=True,
    package_data={
        "hierarchical_rag": ["*.txt", "*.md"],
    },
) 