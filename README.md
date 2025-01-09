# Python Version of the Speechless Reader Model - v0.01

## Deploy Locally

***Install:***
``` bash
pip install -r requirements.txt
```

***Run UI:***
``` bash
python -m streamlit run py_slr_ui/app.py
```

***Run Simulation Code only:***
Input files: "data" folder
Output files: "output" folder
And then run the command:
``` bash
python -m main
```


## Deploy on Heroku

Sign up & Download Heroku CLI

***Clone Project and Create App:***
``` bash
heroku login
git clone https://github.com/bgagl/py_slr_v.01.git
cd py_slr_v.01
heroku create lex-app
```

***Deploy:***
``` bash
git add .
git commit -m "Initial commit"
git push heroku master
```


## Commit Changes 
If you want to push code changes to GitHub, you would use:
``` bash
git push origin master
```
