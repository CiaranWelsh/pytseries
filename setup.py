from distutils.core import setup, Extension
from pytseries import __version__

setup(name='pytseries',
      version=__version__,
      description='Python package for handling time series data',
      author='Ciaran Welsh',
      author_email='c.welsh2@newcastle.ac.uk',
      packages=['pytseries'],
      url=r'https://github.com/CiaranWelsh/pytseries',
     )
