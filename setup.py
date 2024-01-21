from setuptools import find_packages,setup
from typing import List

def get_requirement(file_path:str)->List[str]:
    requirement = []
    with open(file_path) as file_obj:
        requirement = file_obj.readlines()
        requirement = [req.replace('\n',"") for req in requirement]

    return requirement

setup(
    name='Captcha Detection',
    version='0.0.1',
    author='Yashkumar Dubey',
    author_email="ydubey7979@gmail.com",
    packages=find_packages(),
    install_requires = get_requirement('requirements.txt')
)