import setuptools 

VERSION = '0.0.1' 
DESCRIPTION = 'Bayesian Inferences'
LONG_DESCRIPTION = 'Run Bayesian inferences models'

# Setting up
setuptools.setup(
       # the name must match the folder name 'verysimplemodule'
        name="BI", 
        version=VERSION,
        author="Sebastian Sosa",
        author_email="<s.sosa@live.fr>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=setuptools.find_packages(),
        install_requires=['numpyro', 'pandas', 'seaborn', 'tensorflow'],
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