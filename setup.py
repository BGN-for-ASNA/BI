import setuptools 
import os
VERSION = '0.0.12' 

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Setting up
setuptools.setup(
        name="BayesInference", 
        version=VERSION,
        author="Sebastian Sosa",
        description="GNU GENERAL PUBLIC LICENSE",
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(),
        install_requires=['jax', 'numpyro', 'pandas', 'seaborn', 'tensorflow_probability', 'arviz', 'funsor'],
        extras_require={
            "cpu": ["jax[cpu]"],
            "cuda12": ["jax[cuda12]"]},
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
            "Operating System :: POSIX :: Linux",
        ],
        project_urls={
            "Homepage": "https://github.com/BGN-for-ASNA/BI",
            "Bug Tracker": "https://github.com/BGN-for-ASNA/BI/issues"
        }
)