from setuptools import setup, find_packages

setup(
    name='bayesian-flow-torch',
    packages=find_packages(),
    version='0.0.1',
    license='MIT',
    description='Bayesian Flow Networks - Pytorch',
    long_description_content_type='text/markdown',
    author='Thorin Farnsworth',
    author_email='thorin.j.a.farnsworth@gmail.com',
    url='https://github.com/thorinf/bayesian-flow-pytorch',
    keywords=[
        'artificial intelligence',
        'deep learning',
        'generative ai'
        'bayesian flow networks',
        'bayesian flow'
    ],
    install_requires=[
        'torch>=1.6'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)
