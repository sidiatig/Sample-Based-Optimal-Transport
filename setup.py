from setuptools import setup, find_packages

setup(
    name='sbot',
    version='0.1',
    description='Sample based optimal transport solver',
    long_description=open('README.md').read(),
    url='https://github.com/monty47/sbot',
    author='Monty Essid',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='optimal transport data sample pointcloud', 
    packages=['sbot'],
    install_requires=['numpy','scipy'],
    zip_safe=False
)
