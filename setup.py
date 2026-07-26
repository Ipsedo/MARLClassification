from setuptools import setup

setup(
    name="marl_classification",
    version="1.2.0",
    author="Samuel Berrien",
    url="https://github.com/Ipsedo/MARLClassification",
    packages=[
        "marl_classification",
        "marl_classification.data",
        "marl_classification.core",
        "marl_classification.networks",
        "marl_classification.training",
    ],
)
