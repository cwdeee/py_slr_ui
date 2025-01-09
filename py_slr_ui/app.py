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

from streamlit.components.v1 import html


function_dict = {
    "Version 2 (Current Version)": "py_slr_v.01-main_rel2",
    "Version 1": "py_slr_v.01-main_rel1"
}



cd = str(Path(__file__).parent.parent.joinpath("py_slr_ui").resolve()) #str(os.getcwd())#str(Path(__file__).parent) #os.getcwd() 
#st.write(cd)
folder_in = cd+"/input_data/"
folder_in2 = cd+"/template_data/"
folder_out = cd+"/output_data/"


# Set the app configuration
st.set_page_config(page_title="Self Learning Systems Lab", layout="centered")


# Add a banner at the top
st.markdown(
    """
    <div style="
        background-color: #D4EDDA; 
        padding: 10px; 
        border-radius: 5px; 
        text-align: center; 
        font-weight: bold; 
        font-size: 16px;
        color: #155724;
    ">
        ⚠️ This page is just showing dummy results for demonstration purposes.
    </div>
    """,
    unsafe_allow_html=True
)
# st.warning("⚠️ This page is just showing dummy results for demonstration purposes.")

# Add the logo to the app
st.image(str(Path(__file__).parent.parent.joinpath("py_slr_ui","logo.png").resolve()), width=200)
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
    console.log("running simulation ...")
    #run_simulation(stimuli_words, stimuli_non_words, lexicon)
# Run button
if st.button("Run"):
    # Display "Calculation started" message
    status_message = st.empty()
    status_message.write("Calculating using algorithm " + selected_algorithm_name + " ...")

    # Start the long computation in a separate thread to avoid blocking
    computation_thread = threading.Thread(target=long_computation)
    computation_thread.start()

    # Simulate the progress bar while the real computation runs
    simulate_progress_bar(3)

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
    st.table(df)