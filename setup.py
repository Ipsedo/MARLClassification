from setuptools import setup

setup(
    name="marl_classification",
    version="1.0.0",
    author="Samuel Berrien",
    url="https://github.com/Ipsedo/MARLClassification",
    packages=[
        "marl_classification",
        "marl_classification.data",
        "marl_classification.environment",
        "marl_classification.networks"
    ],
    license='GPL-3.0 License'
)
