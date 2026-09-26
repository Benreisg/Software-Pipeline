from setuptools import setup, find_packages

setup(
    name='bef4llm',
    version='1.0',
    url='git@gitlab-iwi.dfki.de:lauer/bef4llm.git',
    python_requires='>=3.11',
    install_requires=[
        'setuptools',
    ],
    packages=find_packages("src"),
    package_dir={"": "src"},
    zip_safe=False
)
