from setuptools import setup

with open('requirements.txt', 'r') as f:
    requires = list(f.readlines())

setup(name='st',
      version='0.1',
      description='Stencil code generator',
      author='Tuowen Zhao',
      author_email='ztuowen@gmail.com',
      license='MIT',
      packages=['st', 'st.codegen', 'st.codegen.backend'],
      scripts=['vecscatter'],
      python_requires='>=3.6.0',
      install_requires=requires,
      zip_safe=False)
