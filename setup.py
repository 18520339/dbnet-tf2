import setuptools

setuptools.setup(
    name = 'tfdbnet',
    version = '0.0.3',
    author = 'Quan Dang',
    author_email = '18520339@gm.uit.edu.vn',
    description = 'A TensorFlow 2 wrapper of DBNet',
    long_description = open('README.md', 'r').read(),
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/18520339/dbnet-tf2',
    packages = setuptools.find_packages(),
    install_requires = [
        'imagesize==1.3.0',
        'keras_resnet==0.2.0',
        'numpy==1.22.3',
        'opencv_python_headless==4.5.5.64',
        'pyclipper==1.3.0.post2',
        'scipy==1.7.3',
        'Shapely==1.8.1.post1',
        'tensorflow==2.9.0',
        'tqdm==4.64.0',
    ],
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)