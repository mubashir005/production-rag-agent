from setuptools import setup, find_packages

setup(
    name="nvidia-rag-agent",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
    "console_scripts": [
        "rag=app.cli:main",
    ]
},

)
