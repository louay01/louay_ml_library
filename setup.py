from setuptools import setup, find_packages
from pathlib import Path

ROOT = Path(__file__).resolve().parent

classifiers = [
    'Development Status :: 4 - Beta',          
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Operating System :: OS Independent',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3', 
]

setup(
    name='louay_ml_library',
    version='0.0.4',
    author='Louay El Masri',
    description='A custom machine learning library containing several models',
    long_description=(ROOT / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    url='https://github.com/louay01/louay_ml_library',
    packages=find_packages(include=['louay_ml_library', 'louay_ml_library.*'],
                           exclude=['louay_ml_library.tests', 'louay_ml_library.tests.*']),
    keywords='machine learning linear regression linear classifier neural networks decision trees SVM AI',
    license='MIT',
    classifiers=classifiers,
    python_requires='>=3.10',
    install_requires=['numpy>=1.23'],
    extras_require={'metrics': ['scikit-learn']},
)
