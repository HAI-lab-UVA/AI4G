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

## Generating Feedback

Generating feedback consists of a few steps.
1. Survey spreadsheet is generated using demand and allocation data.
1. Survey is uploaded to google drive for distribution.
1. Completed survey is downloaded.
1. Survey responses are split and converted into Allocation feedback and Item feedback.
1. Missing feedback is predicted.

### Generating Survey spreadsheet
Both demand and allocation decision data must be present to generate the spreadsheet.

Demand csv file should be placed under `comms2/Demand` with the filename `Predicted_Demand<DAY_NUM>.csv`

Allocation decision file is found in `decision/Data/Part2` with the filename format `alloation_day<DAY_NUM>.csv`

The script to generate the spreadsheet is run using the following commmand.
```
$ python feedback-spreadsheet.py
Enter the feedback day to generate spreadsheet: <DAY_NUM>
```
When prompted, enter the `<DAY_NUM>` corresponding to the filenames

The generated file will be found under `comms2/Feedback_Responses` in the format `feedback_day<DAY_NUM>.csv`

### Uploading/Downloading Survey spreadsheet to google drive
Generated survey spreadsheets can be automatically uploaded to google drive for distribution to users. This is done using the following script.
```
$ python upload_download_responses.py
Type 'upload' to upload generated feedback file or 'download' to download the file: 
Enter the feedback day to upload: 
```
When prompted, enter `upload` or `download`, and the `<DAY_NUM>`.

When downloading, the feedback files will be overwritten with user response data. These files will be found under the same directory, `comms2/Feedback_Responses` in the same file format, `feedback_day<DAY_NUM>.csv`.

### Generating Allocation and Item feedback
The survey spreadsheet can be split into Allocation and Item feedback csv files using the following script.
```
$ python handle_user_feedback.py
Enter the feedback day: 
```
When prompted, enter the `<DAY_NUM>`.

The files will be saved under `comms2/Separate_Responses` with the file format `allocation_feedback_day<DAY_NUM>.csv` and `pref_feedback_day<DAY_NUM>.csv`.

### Running the Feedback Module

The feedback module can be run in the following way

```
$ python main.py <DECISION_CSV> <ALLOC_FEEDBACK_CSV> <ITEM_FEEDBACK_CSV> <USER_SIMILARITY_CSV> <ITEM_SIMILARITY_CSV> <DECISION_HISTORY_CSV> <DAY_NUM>
```

Output files will be saved under `comms2/OUTPUT` with the file format `item_feedback_day_<DAY_NUM>.csv` for item feedback and `predicted_allocation_feedback_<DAY_NUM>.csv` for allocation feedback.

