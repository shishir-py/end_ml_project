from setuptools import find_packages, setup
from typing import List
hyphen_e ='-e .'
def get_requires(file_path:str) ->List[str]:
        '''
        List out list of requirements that will be needed
        '''
        requirements=[]
        with open('requirements.txt') as file_obj:
            requirements=file_obj.readlines()
            requirements=[req.replace("\n",' ')for req in requirements]
            if hyphen_e in requirements:
                requirements.remove(hyphen_e)
        return requirements
setup(
    name="ml_project",
    version="0.0.1",
    author="shishir",
    author_email="sheahead22@gmail.com",
    packages=find_packages(),
    install_requires=get_requires('requirements.txt')
    
)