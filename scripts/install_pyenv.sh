#!/bin/bash
append_if_not_exists() {
    if grep -q "$1" $(realpath ~/.bashrc); then
        echo "$1" already exsits.
    else
        echo "$1" >> ~/.bashrc      
    fi
}

curl https://pyenv.run | bash
append_if_not_exists 'export PYENV_ROOT="$HOME/.pyenv"'
append_if_not_exists 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"'
append_if_not_exists 'eval "$(pyenv init -)"'
append_if_not_exists 'eval "$(pyenv virtualenv-init -)"'
exec "$SHELL"