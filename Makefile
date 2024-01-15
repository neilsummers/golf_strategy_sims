.PHONY: clean setup venv

setup: venv/bin/python

clean:
	rm -rf venv

venv: clean venv/bin/python

venv/bin/python:
	virtualenv -p python3.10 venv
	. venv/bin/activate ;\
	pip install -U pip setuptools wheel ;\
	pip install -r requirements.txt
