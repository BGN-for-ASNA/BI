from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Bayesian Inferences'
LONG_DESCRIPTION = 'Run Bayesian inferences models'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="BI", 
        version=VERSION,
        author="Sebastian Sosa",
        author_email="<s.sosa@live.fr>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        keywords=['python', 'Bayesian inferences'],
        include_package_data=True,
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)