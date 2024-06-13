all: \
	lock_dependencies \
	format_data \
	build_dependencies \
	create_plots \
	eda \
	experiments \
	jupyter_notebook

# Setup workflow. Execute targets individually
setup: \
	install_python \
	create_virtual_env \
	build_dependencies \
	install_requirements

setup_environment: \
	create_virtual_env \
	build_dependencies \
	install_requirements

install_python:
	@sudo apt update && \
	sudo apt-get install -y python3 python3-pip && \
	sudo apt install -y build-essential libssl-dev zlib1g-dev \
		libbz2-dev libreadline-dev libsqlite3-dev curl \
		libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev && \
	./scripts/install_pyenv.sh

create_virtual_env:
	@pyenv install 3.12.3 && \
	pyenv virtualenv 3.12.3 learn-ml && \
	pyenv local learn-ml

build_dependencies:
	@python -m build modules

install_requirements:
	@pip install -r requirements.txt && \
	pip install modules/dist/*.tar.gz

clean:
	@rm visualization/outputs/* 2>&1 > /dev/null | true

lock_dependencies:
	@pip-compile requirements.in

format_data:
	@python format_data.py

jupyter_notebook:
	@jupyter notebook --no-browser

create_plots: \
	clean
	python create_plots.py

experiments: \
	python ./experiments/multilinear_interpoloation.py && \
	python ./experiments/standardScaling_classification.py && \
	python ./experiments/standardScaling_regression.py

eda:
	python eda.py