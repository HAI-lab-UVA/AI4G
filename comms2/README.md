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

## Generarting Feedback

Generating feedback consists of two steps.
* First, survey responses must be split and converted into Allocation feedback and Item feedback. This is done through the `handle_user_feedback.py` script
* Missing feedback is predicted. This is done through the feedback module

### Generating Allocation and Item feedback
```
$ python handle_user_feedback.py
```


### Running the Feedback Module

The feedback module can be run either from the command line or imported into a python program as a module

#### Command line
```
$ python main.py <DECISION_CSV> <ALLOC_FEEDBACK_CSV> <ITEM_FEEDBACK_CSV> <USER_SIMILARITY_CSV> <ITEM_SIMILARITY_CSV>
```

#### Module
```
from feedback_module import Feedback
    
f = Feedback()

# generate predicted feedback df
feedback_prediction_df = f.predict_feedback(
    <DECISION_DF>,
    <ALLOC_FEEDBACK_DF>,
    <ITEM_FEEDBACK_DF>,
    <USER_SIMILARITY_DF>,
    <ITEM_SIMILARITY_DF>
)

# get raw item feedback data
f.get_item_feedback()
```
