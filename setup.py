import setuptools 

VERSION = '0.0.1' 
DESCRIPTION = 'Bayesian Inferences'
LONG_DESCRIPTION = 'Run Bayesian inferences models'

# Setting up
setuptools.setup(
        name="BI", 
        version=VERSION,
        author="Sebastian Sosa",
        author_email="<s.sosa@live.fr>",
        description=MIT License,
        long_description="Bayesian Estimation Environment (BEE) software is availble in both Python and R. It aims to unify the modeling experience by integrating an intuitive model-building syntax with the flexibility of low-level abstraction coding available but also pre-build function for high-level of abstraction and including hardware-accelerated computation for improved scalability.",
        packages=setuptools.find_packages(),
        install_requires=['jax[cuda12]', 'numpyro', 'pandas', 'seaborn', 'tensorflow_probability', 'arviz'],
        python_requires=">=3.9",
        keywords=['python', 'Bayesian inferences'],
        include_package_data=True,
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: Microsoft :: Linux",
        ],
        project_urls={
            url = "https://github.com/BGN-for-ASNA/BI",
            url = "https://github.com/BGN-for-ASNA/BI/issues"
        }
)