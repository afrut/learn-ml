# sudo apt install make

PYTHON_VERSION=3.10.16
VENV_NAME=learn-ml

all: \
	lock_dependencies \
	format_data \
	build_and_install_modules \
	create_plots \
	eda \
	experiments \
	jupyter_notebook

# Setup workflow. Execute targets individually
setup: \
	install_python \
	install_pyenv \
	create_venv \
	install_requirements \
	build_and_install_modules \
	init_dirs \
	kaggle_symlinks

install_python:
	@sudo apt update && \
	sudo apt-get install -y python3 python3-pip && \
	sudo apt install -y build-essential libssl-dev zlib1g-dev \
		libbz2-dev libreadline-dev libsqlite3-dev curl \
		libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

install_pyenv:
	./scripts/install_pyenv.sh

create_venv:
	@pyenv install ${PYTHON_VERSION} --skip-existing && \
	pyenv uninstall --force ${VENV_NAME} && \
	pyenv virtualenv ${PYTHON_VERSION} ${VENV_NAME} -f && \
	pyenv local ${VENV_NAME}

install_requirements:
	@pip install -r requirements.txt

lock_dependencies:
	@pip-compile requirements.in

build_and_install_modules: build_modules install_modules

build_modules:
	@python -m build modules

install_modules:
	@pip install modules/dist/*.whl --force-reinstall

init_dirs:
	@sudo mkdir -p /kaggle/input

clean:
	@rm visualization/outputs/* 2>&1 > /dev/null | true


format_data:
	@python format_data.py

kaggle_symlinks:
	@sudo rm -rf /kaggle/input/titanic && sudo ln -s $(realpath ./kaggle/titanic/data) /kaggle/input/titanic && \
	sudo rm -rf /kaggle/input/home-data-for-ml-course && sudo ln -s $(realpath ./kaggle/home-data-for-ml-course/data) /kaggle/input/home-data-for-ml-course

jupyter_notebook:
	@jupyter notebook --no-browser

jupyter_lab:
	@jupyter lab --no-browser \
		--NotebookApp.token="abcd"

create_plots: \
	clean
	python create_plots.py

experiments: \
	python ./experiments/multilinear_interpoloation.py && \
	python ./experiments/standardScaling_classification.py && \
	python ./experiments/standardScaling_regression.py

eda:
	python eda.py