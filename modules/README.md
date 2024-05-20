# Building and Packaging
- When packaging python projects, it is conventional to put all source code in a
  *src* directory.
- The following files are added to standardize the structure of the project.
  - LICENSE
  - pyproject.toml - The project configuration file
  - README.md
  - tests - Contains testing-related files
- Install build frontend
  ```
  pip install --upgrade build
  ```
- Execute build to package the project.
  - dist/*.tar.gz is a source distribution.
  - dist/*.whl is a built/binary distribution.
  ```
  python -m build
  ```
- The resulting files can now be uploaded to PyPi or installed in other
  environments.
- Create and activate a virtual environment.
  ```
  python -m venv env
  source env/bin/activate
  ```
- Test installing via source distribution.
  ```
  pip install dist/project_name_112bf10294ca91e-0.0.1.tar.gz
  ```
- Uninstall.
  ```
  pip uninstall -y project-name-112bf10294ca91e
  ```
- Test installing via built distribution.
  - Built distributions install faster.
  ```
  pip install dist/project_name_112bf10294ca91e-0.0.1-py3-none-any.whl
  ```

# Installation
- Install and update tools that manage python packages.
  - *pip* is for installing packages.
  - *setuptools* is used fro installing from source code.
  - *wheels* are pre-built binary package format for modules and libraries.
  - The following command will install these packages to the system-wide
    *site-packages* directory.
    - On Windows, it is located in */path/to/python/installation/Lib/site-packages*.
  ```
  python3 -m pip install --upgrade pip setuptools wheel
  ```
- User installation of a package to avoid breaking system-wide packages.
  Installs to a user-specific location.
  - On Windows, installs to *C:\Users\user\appdata\roaming\python\pythonVersion\site-packages*.
  ```
  pip install --user some_package
  ```
- Install a package from local source in development mode.
  ```
  pip install -e <path_to_source_code>
  ```
- See *devtools_envs* for dependency management and environment setup.

# Links
- [Packaging Guide](https://packaging.python.org/en/latest/)
- [Packaging Overview](https://packaging.python.org/en/latest/overview/)
- [Installation guide](https://packaging.python.org/en/latest/tutorials/installing-packages/)
- [Installing standalone command line tools](https://packaging.python.org/en/latest/guides/installing-stand-alone-command-line-tools/)
- [Managing dependencies](https://packaging.python.org/en/latest/tutorials/managing-dependencies/#managing-dependencies)
- [Packaging Python projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/#)
  - [Declaring project metadata](https://packaging.python.org/en/latest/specifications/declaring-project-metadata/#declaring-project-metadata)
  - [Choosing a build backend](https://packaging.python.org/en/latest/tutorials/packaging-projects/#choosing-a-build-backend)
- [Packaging and distributing projects](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#distributing-packages)
- [`pip`'s Build System Interface](https://pip.pypa.io/en/stable/reference/build-system/)