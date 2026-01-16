from setuptools import find_packages, setup

setup(
    name="ordinalclip",
    packages=find_packages() + ["scripts", "configs"],
    package_data={
        "configs": ["*.yaml"],
        "scripts": ["*.py"],
    },
    include_package_data=True,
)
