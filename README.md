# Setup

- [Install `pyenv`](https://github.com/pyenv/pyenv?tab=readme-ov-file#getting-pyenv)
- [Install `pyenv-virtualenv`](https://github.com/pyenv/pyenv?tab=readme-ov-file#getting-pyenv)
- Install Python and create virtual environment

  ```
  sudo apt-get install \
    libbz2-dev \
    libssl-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline6-dev \
    libffi-dev \
    libsqlite3-dev \
    liblzma-dev
  pyenv install 3.11.8
  pyenv virtualenv 3.11.8 learn-ml
  pyenv local learn-ml
  make lock_dependencies
  ```
