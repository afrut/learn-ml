all: \
	lock_dependencies \
	format_data \
	jupyter_notebook

lock_dependencies:
	pip-compile requirements.in

format_data:
	python data/format_data.py

jupyter_notebook:
	jupyter notebook --no-browser &