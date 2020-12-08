from setuptools import setup

#from distutils.core import setup
import setuptools

import os


def read_requirements():
    '''parses requirements from requirements.txt'''
    reqs_path = os.path.join('.', 'requirements.txt')
    install_reqs = parse_requirements(reqs_path, session=PipSession())
    reqs = [str(ir.req) for ir in install_reqs]
    return reqs

setup(
    name='jithub',
    version='0.1.0',
    description='A example Python package',
    url='https://github.com/russelljjarvis/jit-hub',
    author='Russell Jarvis',
    author_email='russelljarvis@protonmail.com',
    packages = setuptools.find_packages(),
)
