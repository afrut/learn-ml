all: \
	lock_dependencies \
	format_data \
	jupyter_notebook

setup: \
	install_python \
	create_virtual_env \
	install_requirements

install_python:
	@sudo apt update && \
	sudo apt-get install python3 && \
	sudo apt-get install python3-pip && \
	sudo apt install -y build-essential libssl-dev zlib1g-dev \
		libbz2-dev libreadline-dev libsqlite3-dev curl \
		libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev && \

create_virtual_env:
	@pyenv install 3.12.3 && \
	pyenv virtualenv 3.12.3 learn-ml && \
	pyenv local learn-ml

install_requirements:
	pip install -r requirements.txt

clean:
	@rm visualization/outputs/* 2>&1 > /dev/null | true

lock_dependencies:
	pip-compile requirements.in

format_data:
	python data/format_data.py

jupyter_notebook:
	jupyter notebook --no-browser &

plots: \
	clean
	@DISPLAY=:0 \
	PYTHONPATH=$$(realpath "modules") \
	python visualization/visualization.py