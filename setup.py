from setuptools import setup, find_packages

# desriptions
with open("README.md", 'r') as f:
    long_description = f.read()
short_description = 'Thin Section Analysis toolset for segmenting' + \
                    'then subsequenly analysing Thin Section Images'
    
setup(
   name='TSA',
   version='1.0.0',
   description='Thin Section Analysis toolset for segmenting then subsequenly analysing Thin Section Images',
   license="MIT",
   long_description=long_description,
   author='Richard Boyne',
   author_email='rmb115@ic.ac.uk',
   url="https://github.com/msc-acse/acse-9-independent-research-project-Boyne272",
   packages=find_packages(),
#    packages=['TSA'],
#    packages=['TSA.kmeans', 'TSA.merging', 'TSA.pre_post_processing', 'TSA.tools'],
   install_requires=['numpy', 'matplotlib', 'scipy', 'torch', 'scikit-image']
)

print(find_packages(), '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')