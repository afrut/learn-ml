# Setup

- [Install `pyenv`](https://github.com/pyenv/pyenv?tab=readme-ov-file#getting-pyenv)
- [Install `pyenv-virtualenv`](https://github.com/pyenv/pyenv?tab=readme-ov-file#getting-pyenv)
- Install Python and create virtual environment

  ```
  sudo apt update
  sudo apt-get install python3
  sudo apt-get install python3-pip
  sudo apt install build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
  pyenv install 3.12.3
  pyenv virtualenv 3.12.3 learn-ml
  pyenv local learn-ml
  pip install -r requirements.txt
  ```
