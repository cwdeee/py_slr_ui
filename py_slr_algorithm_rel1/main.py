import numpy as np
import pandas as pd
from pandas import read_csv
from pathlib import Path

from helpers.helper_functions import wrapper_lexicon_wide_pe_estimation, initiate_lexicon, wrapper_store_oPE_perc, \
    wrapper_lexicon_wide_pe_estimation_preview, return_df_preview


def main(base_dir):
    cd = str(base_dir)
    folder_in = cd+"/input_data"
    folder_in2 = cd+"/template_data"
    folder_out = cd+"/output_data"

    human_lex = initiate_lexicon(word_file_task=folder_in + '/human_w.csv',
                                 word_file_lexicon=folder_in + '/human_w_lexicon.csv',
                                 non_word_task=folder_in + '/human_non-words.csv')
    
    is_calc2 = False

    threshold_version = 5
    if threshold_version == 1:
        is_calc2 = True
    elif threshold_version == 2:
        from numpy import arange
        thresholds = arange(60, 70, 30) #[0.07]
        is_boundary = False
    elif threshold_version == 3:
        thresholds = [0.7]
        is_boundary = True
    elif threshold_version == 4:
        threshold_df = read_csv(folder_in + '/human_thresholds.csv', header=None)
        thresholds = threshold_df[threshold_df.columns.to_list()[0]].tolist()
        
        if len(thresholds)>0:
            is_boundary = True
        else:
            is_boundary = False

        # Weil von Kommawerten ausgegangen wird
        is_boundary = True
    elif threshold_version == 5:
        print("hi")
        with open(folder_in + '/human_thresholds.csv', 'r', encoding='utf-8') as file:
            content = file.read()
        #thresholds = content.split("\n")
    
        # Split into lines and convert each non-empty line to float
        thresholds = [float(x) for x in content.split("\n") if x.strip() != ""]
        print(thresholds)
        # if all values are above 
        is_boundary = not all(value > 1 for value in thresholds) 
        print(is_boundary)
        if not is_boundary:
            print("Numbers instead of Decimal")
            thresholds = [int(x) for x in thresholds]
            print(thresholds)
        #is_boundary = True

    if is_calc2 == True:
        wrapper_lexicon_wide_pe_estimation(human_lex, file_name=folder_out+"/results.csv", ope_version="gagl_2020", word_in_lex=True)
    else:
        print(thresholds)
        wrapper_lexicon_wide_pe_estimation(human_lex, file_name=folder_out+"/results.csv", ope_version="gagl_2020", word_in_lex=True, dec_boundary=is_boundary,boundaries=thresholds)

    results = read_csv(str(base_dir)+"/output_data/results.csv", header=None)
    print(results)

if __name__ == '__main__':
    base_dir = Path(__file__).parent.parent.resolve() 
    main(base_dir)
    results = read_csv(str(base_dir)+"/output_data/results.csv", header=None)
    print(results)
