# Communications Group 2: Feedback

Members:
Yasunari Kato (yk2f)
Courtney Kennedy (csk6snj)
Alex Kwakye (ak3gj)

## Setting Up the Environment

This tool has been tested on Python 3.8


### Optional: Setting up a virtual environment
Virtual environments allow for the separation of python packages based on the current working environment. It also allows for the concurrent use of different python versions.

More information can be found [here](https://docs.python.org/3.8/library/venv.html).

```
$ python -m venv venv
$ source venv/bin/activate
```

### Install all dependencies
```
$ pip install -r requirements.txt
```

## Running the Feedback Module

The feedback module can be run either from the command line or imported into a python program as a module

### Command line
```
$ python main.py
```

### Module
```
from feedback_module import Feedback
    
f = Feedback()
```
