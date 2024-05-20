all: \
	setup \
	build \

setup: \
	create_virtual_env \
	install_dependencies

create_virtual_env:
	@pyenv virtualenv 3.12.3 learn-ml-modules | true && \
	pyenv local

install_dependencies:
	pip install -r requirements.txt

build:
	python -m build