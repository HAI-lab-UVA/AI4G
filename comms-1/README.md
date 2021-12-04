# AI4G

Necessary steps to run the code:

Install Surprise

To do this you can follow the instructions on their page (https://surpriselib.com/ or https://github.com/NicolasHug/Surprise)

If you want a tl;dr:
* If you have conda use:
  * ```
    $ conda install -c conda-forge scikit-surprise
    ```
* If just pip use:
  * ```
    $ pip install numpy
    $ pip install scikit-surprise
    ```
* You may need to also install c++ version 14, but if you run into this issue, message either Andrew or Austin on slack.

Other dependencies
  You will need _pandas_, _numpy_, and _sklearn_, though these processes are much more straight forward

Running the code
  To verify that the code is running correctly, move to the comms1 folder and execute test.py:
  ```
  $ python test.py
  ```
  or:
  ```
  $ python3 test.py
  ```
You should get something similar to this (don't worry about the decimal places being slightly different): ![image](https://user-images.githubusercontent.com/42854353/144130430-b5aa09ed-d059-49ff-a1fd-0a1eb73bf44d.png)

At this point you will be within the python debugger.
Possible commands include:
  * To exit, type "exit" and hit enter.
  * To get user similarities, type `algo.sim_mat(is_user = True)`
  * To get item similarities, type `algo.sim_mat(is_user = False)`
  * To get \hat{R} (The user by item matrix with all the ratings), type `algo.construct_RHat()
  *  Otherwise, generic python commands will run (for more on the python debugger explore here: https://docs.python.org/3/library/pdb.html

To import our methods into your code you will need the following:
```
from preference import SVDpp_neighborhood
path = "data\\user_responses.csv"   # or whatever your path ends up as

algo = SVDpp_neighborhood(data_path = path, verbose = False)
algo.fit(n_epochs = 5)
```

From here you can call any of the following SVDpp_neighborhood class methods
* algo.fit(n_epochs = 20)
* algo.estimate(u, i)
* algo.sim_mat(is_user = True, as_df = False, write_out = False)
* algo.construct_RHat(df = True, write_out = False, OVERRIDE = -10000)
* algo.data.drop(columns = algo.pref.columns)           # for just demographical information
* algo.pref                                             # for just preferences information
* algo.data                                             # for combined preference and demographical data
* algo.user_mapping
* algo.item_mapping

Others will be added to the list upon completion (notably adding new users and new items) (and new features are semi-available upon request)

Let us know if you have any questions!
