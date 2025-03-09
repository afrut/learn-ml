#!/bin/bash
# Also installs pyenv-virtualenv

append_if_not_exists() {
    path=$(realpath "$2")
    # Search for a line in file
    if grep -q "$1" $path; then
        echo "$1" already exsits.
    else
        # echo "    ${1}, $path"
        echo "$1" >> $path
    fi
}

curl https://pyenv.run | bash

lines=(\
    '# For managing python versions and virtualenvs'
    'export PYENV_ROOT="$HOME/.pyenv"'\
    '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"'\
    'eval "$(pyenv init - bash)"'
)

bash_files=(\
    $(realpath ~/.bashrc)\
    $(realpath ~/.profile)\
    $(realpath ~/.bash_profile)\
    $(realpath ~/.bash_login)\
)

for bash_file in "${bash_files[@]}"
do
    if [ -f $bash_file ]; then
        # echo $bash_file
        echo "" >> $bash_file
        for line in "${lines[@]}"
        do
            append_if_not_exists "$line" $bash_file
        done
    fi
done
append_if_not_exists 'eval "$(pyenv virtualenv-init -)"' $(realpath ~/.bashrc)
source ~/.bashrc