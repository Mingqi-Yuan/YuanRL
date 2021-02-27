import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="yuanrl", # Replace with your own username
    version="0.0.3",
    author="Mingqi Yuan",
    author_email="friedrichyuan19990827@gmail.com",
    description="Implementations of the Single-Agent Reinforcement Learning (SARL) and Multi-Agent Reinforcement Learning (MARL) algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mingqi-Yuan/YuanRL",
    project_urls={
        "Bug Tracker": "https://github.com/Mingqi-Yuan/YuanRL/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)