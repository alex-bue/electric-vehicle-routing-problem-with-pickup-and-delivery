from setuptools import setup, find_packages

setup(
    name='evrp',
    version='0.1.0',
    python_requires=">=3.9, <3.11",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "evrp": ["data/*.csv", "data/*.json"]
    },
    entry_points={
        'console_scripts': [
            'vrp-solver=evrp.__main__:main',
        ],
    },
)
