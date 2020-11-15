from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='RVEstimator',
      version='0.1',
      description='Python Tool for Estimating RV from spectrum',
      long_description = readme(),
      classifiers=[
          'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering :: Astronomy',
      ],
      keywords='Radial Velocity Astronomy Spectrum',
      url='https://github.com/indiajoe/RVEstimator',
      author='Joe Ninan',
      author_email='indiajoe@gmail.com',
      license='LGPLv3+',
      packages=['RVEstimator'],
      entry_points = {
          'console_scripts': ['calculate_rv=RVEstimator.calculate_rv:main'],
      },
      install_requires=[
          'numpy',
          'scipy',
          'astropy',
          'pandas',
          'functools32;python_version<"3"',
      ],
      include_package_data=True,
      zip_safe=False)
