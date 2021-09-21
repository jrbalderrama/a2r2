# Atelier d'Anonymisation - Les Rencontres RUDI 2021

## Install
This project runs on a Jupyter server compatible with Python 3.8.
If you run the project on [colab](https://colab.research.google.com/),
the following installation is optional (unless you would like to run on 
*colab* using local resources).

In order to run the notebooks without *colab*, first, you might create a
virtual environment (optional). Second, clone the sources and install 
the dependencies. To get them, install `poetry`, and then run the 
`install` command. The complete installation sequence is as follows 
(set the `PATH` accordingly):

```bash
python3.8 -m venv ${PATH}/venv-a2r2
source ${PATH}/venv-a2r2/bin/activate 
git clone https://github.com/jrbalderrama/a2r2.git 
cd a2r2
pip install poetry
poetry install
```

## Run
The project is organized in self-contained Jupyter notebooks.
There are two ways to run them:

- On [colab](https://colab.research.google.com/), click on the links of
  each notebook to get access using your own Google credentials.
- On a local instance of a Jupyter server, after the dependencies 
  installation, launch Jupyter (copy the link displayed in the terminal 
  on a Web browser) and click on a notebook from the list.

  ```bash
  jupyter notebook
  ```
