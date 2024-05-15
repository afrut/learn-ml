lock_dependencies:
	pip-compile requirements.in

jupyter_notebook:
	jupyter notebook --no-browser &