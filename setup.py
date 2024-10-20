from setuptools import setup, find_packages

setup(
    name='minbpe',
    version='0.1',
    packages=find_packages(),
    install_requires=['regex' ,'tiktoken'],  # List your package dependencies here
)