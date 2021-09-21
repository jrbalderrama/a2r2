# a2r2 : Atelier d'Anonymisation - Les Rencontres RUDI 2021

## Install
This project runs on a Jupyter server installed on Python 3.8. First, you can 
create a virtual environment. Second, clone the sources and install the
dependencies on the virtual environment. In order to get them, install 
`poetry`, and then run the install command. The complete installation 
sequence is as follows (set the `PATH` accordingly):

```bash
python3.8 -m venv ${PATH}/venv-a2r2
source ${PATH}/venv-a2r2/bin/activate 
git clone https://github.com/jrbalderrama/a2r2.git 
cd a2r2
pip install poetry
poetry install
```

