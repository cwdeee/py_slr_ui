# py_slr_v.01-ui


Folders:
* py_slr_v.01-ui: contains the streamlit app
* py_slr_v.01-main_rel1: a downloaded release version (release 1) of the bgagl/py_slr_v.01-main repo 
* py_slr_v.01-main_rel2: a downloaded release version (release 2) of the bgagl/py_slr_v.01-main repo 

Both of those release versions have a main.py file which contains a main() function. 
The "Run"-UI-Button of the streamlit application of py_slr_v.01-ui loads the main.py as a module, depending on which "algorithm version" is selected in the UI, and then runs the main() function. A output file called "results.csv" gets generated, which is displayed in the streamlit UI App as a table. After that the modules loaded get unloaded again. 

In future version it's important that the main.py contains a main() function. 
