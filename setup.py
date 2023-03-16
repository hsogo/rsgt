#!/usr/bin/env python

from setuptools import setup, find_packages
import sys
import os

version = '0.1.0'

setup(name='rsgt',
      version=version,
      description='Online/offline gaze tracking module with a single visible-light camera',
      long_description="""
rsgt is a package for eye movement recording. It detects human face and irises 
from common visible-light camera images (such as web cameras), and records
6D face pose and gaze direction.

Note: due to processing speed issues, real-time recording may result in frame dropping.
The spatial resolution of face and iris images captured by web cameras with typical
view angles is limited, thus heavily constraining the accuracy and precision of the recording.

""",
      classifiers=[
          # http://pypi.python.org/pypi?%3Aaction=list_classifiers
          'Development Status :: 3 - Alpha',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License (GPL)',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
      ],
      keywords='gaze tracking, eye tracking, eye movement',
      author='Hiroyuki Sogo',
      author_email='hsogo12600@gmail.com',
      url='https://github.com/hsogo/rsgt',
      license='GNU GPL',
      install_requires=['numpy', 'scipy', 'matplotlib', 
                        'opencv-python', 'dlib', 'wxPython'],
      packages=['rsgt', 'rsgt.app', 'rsgt.tools',
                'rsgt.iris_detectors'],
      package_data={'rsgt':[
                        'LICENSE.txt',
                        'resources/*.*'],
                    'rsgt.iris_detectors':[
                        '*.dat',
                        'enet/*']
                  },
      #scripts=[]
      )
