import streamlit as st
import time
import numpy as np
import pandas as pd
from pandas import read_csv
import os 
import sys
import importlib
from pathlib import Path
import threading 
import time

import os
import sys
from pathlib import Path

def get_app_base_dir():

    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)

    elif '__file__' in globals():

        return Path(__file__).parent.resolve() #os.chdir(str(Path(__file__).parent.resolve()))

    else:
        return Path.cwd()


try:
    base_dir = get_app_base_dir()
    os.chdir(str(base_dir))
    print(f"[INFO] Working directory changed to: {base_dir}")
except Exception as e:
    print(f"[ERROR] Failed to change directory: {e}")




def load_module_and_run_calculation_function(_dir):
    # Get the path of the module directory
    _path = str(Path(__file__).parent.parent.joinpath(_dir).resolve())
    sys.path.append(_path)
    
    # Record the set of already loaded modules
    before_import = set(sys.modules.keys())
    
    # Load the module and run the calculate command
    print("Using " + _path)
    app_module = importlib.import_module('main')
    os.chdir(_dir)
    app_module.main()
    
    # Unload all new imports (including 'main' and its dependencies)
    after_import = set(sys.modules.keys())
    new_imports = after_import - before_import
    for mod in new_imports:
        del sys.modules[mod]
    
    # Remove the module path from sys.path
    sys.path.remove(_path)

function_dict = {
    "Version 1 (Current Version)": "py_slr_algorithm_rel1"
}




cd = str(Path(__file__).parent.parent.joinpath("py_slr_ui_rel1").resolve()) 
folder_in = cd+"/input_data/"
folder_in2 = cd+"/template_data/"
folder_out = cd+"/output_data/"




# Define the run_simulation function
def run_simulation(stimuli_words, stimuli_non_words, lexicon):


    # Save the inputs into their respective CSV files
    human_w = stimuli_words.split("\n")
    human_non_words = stimuli_non_words.split("\n")
    human_w_lexicon = lexicon.split("\n")
    
    # Convert to DataFrame
    df_human_w = pd.DataFrame(human_w, columns=["Stimuli Words"])
    df_human_non_words = pd.DataFrame(human_non_words, columns=["Stimuli Non-Words"])
    df_human_w_lexicon = pd.DataFrame(human_w_lexicon, columns=["Lexicon"])
    
    # Save to CSV
    df_human_w.dropna().to_csv(folder_in + "human_w.csv", index=False, header=False, encoding='utf-8')
    df_human_non_words.dropna().to_csv(folder_in + "human_non-words.csv", index=False, header=False, encoding='utf-8')
    df_human_w_lexicon.dropna().to_csv(folder_in + "human_w_lexicon.csv", index=False, header=False, encoding='utf-8')
    

    load_module_and_run_calculation_function(function_dict[selected_algorithm_name])
    ##selected_function = function_dict[selected_algorithm_name]
    #st.write("--> "+ selected_algorithm_name)
    #selected_function() # runs simulation code from main.py



# Set the app configuration
st.set_page_config(page_title="Self Learning Systems Lab", layout="centered")

# Add the logo to the app
st.image(str(Path(__file__).parent.parent.joinpath("py_slr_ui_rel1","logo.png").resolve()), width=200)
# Custom theme styling
# Set the light theme in app code
st.markdown("""
    <style>
        .css-18e3th9 {
            background-color: #FFFFFF;  /* White background */
        }
        .css-1d391kg {
            background-color: #F0F8FF;  /* Light blue or secondary background */
        }
        .stTextInput, .stButton {
            color: #1E90FF;  /* Blue color for text and buttons */
        }
        .css-1cpxqw2, .css-1d3xnk6 {
            color: #000000;  /* Black text */
        }
    </style>
    """, unsafe_allow_html=True)

# Set up the Streamlit layout
st.title("SLR v.01 App")


# Initialize text areas with data from CSV files if they exist
def load_csv_to_text_area(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return "\n".join(content.split("\n")[1:])


# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Stimuli Words")
    stimuli_words = st.text_area("Enter Stimuli Words (one per line)", value=load_csv_to_text_area(folder_in2 + "human_w.csv"))

with col2:
    st.subheader("Stimuli Non-Words")
    stimuli_non_words = st.text_area("Enter Stimuli Non-Words (one per line)", value=load_csv_to_text_area(folder_in2 + "human_non-words.csv"))

st.subheader("Lexicon")
lexicon = st.text_area("Enter Lexicon Words (one per line)", value=load_csv_to_text_area(folder_in2 + "human_w_lexicon.csv"))

selected_algorithm_name = st.selectbox("Algorithm Version", options=list(function_dict.keys()))



# Function to simulate progress bar while the computation is running
def simulate_progress_bar(total_time_estimate):
    progress_bar = st.progress(0)
    timer_text = st.empty()
    start_time = time.time()

    for seconds in range(total_time_estimate + 1):
        # Update the progress bar
        progress_bar.progress(int((seconds / total_time_estimate) * 100))

        # Update the elapsed time
        elapsed_time = time.time() - start_time
        timer_text.write(f"Elapsed time: {int(elapsed_time)} seconds")

        # Sleep for 1 second to simulate real-time progress
        time.sleep(1)

        # If the computation is done, break the loop early
        if not computation_thread.is_alive():
            break

    progress_bar.progress(100)  # Ensure the progress bar is 100% at the end


def long_computation(): 
    run_simulation(stimuli_words, stimuli_non_words, lexicon)

# Run button
if st.button("Run"):
    # Display "Calculation started" message
    status_message = st.empty()
    status_message.write("Calculating using algorithm " + selected_algorithm_name + " ...")

    # Start the long computation in a separate thread to avoid blocking
    computation_thread = threading.Thread(target=long_computation)
    computation_thread.start()

    # Simulate the progress bar while the real computation runs
    simulate_progress_bar(20)

    # Wait for the computation to finish
    computation_thread.join()

    # End time after computation is complete
    end_time = time.time()

    # Clear the "Calculation started" message
    status_message.empty()

    # Show the completion message with the actual time
    #st.write(f"Calculation completed.")

    st.success("Simulation completed!")

    df = pd.read_csv(folder_out + "results.csv", index_col=False)
    st.write("### Simulation Results")
    st.dataframe(df, use_container_width=True)
