import numpy as np
import pandas as pd
from pandas import read_csv
from pathlib import Path

from helpers.helper_functions import wrapper_lexicon_wide_pe_estimation, initiate_lexicon, wrapper_store_oPE_perc, \
    wrapper_lexicon_wide_pe_estimation_preview, return_df_preview






def main(base_dir):
    cd = str(base_dir)
    folder_in = cd+"/input_data/"
    folder_in2 = cd+"/template_data/"
    folder_out = cd+"/output_data/"

    human_lex = initiate_lexicon(word_file_task=folder_in + '/human_w.csv',
                                 word_file_lexicon=folder_in + '/human_w_lexicon.csv',
                                 non_word_task=folder_in + '/human_non-words.csv')
    
    threshold_df = read_csv(folder_in + '/human_thresholds.csv')
    thresholds = threshold_df['Thresholds'].tolist()
    if len(thresholds)>0:
        is_boundary = True
    else:
        is_boundary = False
    wrapper_lexicon_wide_pe_estimation(human_lex, file_name=folder_out+"/results.csv", ope_version="gagl_2020", word_in_lex=True, dec_boundary=is_boundary,boundaries=thresholds)



if __name__ == '__main__':
    main()
