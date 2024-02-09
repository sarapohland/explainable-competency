from setuptools import setup, find_packages

requirements = ['numpy',
                'matplotlib',
                'opencv-python',
                'pandas',
                'Pillow',
                'scikit-image',
                'scikit-learn',
                'scipy',
                'seaborn',
                'shapely',
                'torch',
                'torchvision',
                'tqdm'] 

setup(
    name="perception",
    version="1.0.0",
    description="Understanding the Dependence of Perception Model Competency on Regions in an Image",
    packages=find_packages(),
    install_requires=requirements
)