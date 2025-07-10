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

    wrapper_lexicon_wide_pe_estimation(human_lex, file_name=folder_out+"/results.csv", ope_version="gagl_2020", word_in_lex=True)



if __name__ == '__main__':
    main()

def main_old():
    print("### Simulate - static data ###")
    folder_in = "./input_data/parafoveal_preprocessing_stims/"  #"/Users/bg/PycharmProjects/SLR_Model/stimuli_human/"
    folder_out = "./output_data/"  #"/Users/bg/PycharmProjects/py_slr_v.01/simulation_output_files/"

    bsa = 6
    ##################### No PREVIEW oPE standard-fixed lex estimation
    #stims_prepro = read_csv(folder_in + "wordlist_for_Benjamin.txt")
    #stims_prepro.loc[(stims_prepro['nlett'] == bsa), ['string']].to_csv(folder_in + 'temp_files/human_w.csv', index=False,
    #    header=False)


    #lex_prepro = read_csv(folder_in + "lexicon_upper_case_all_lengths.csv")

    #lex_prepro.loc[(lex_prepro['bsa'] == bsa), ['word']].to_csv(folder_in + 'temp_files/human_w_lexicon.csv', index=False,
    #    header=False)  # lex_prepro.loc[lex_prepro['bsa'] == bsa, ['word']]

    #human_lex = initiate_lexicon(word_file_task=folder_in + 'temp_files/human_w.csv',
     #                            word_file_lexicon=folder_in + 'temp_files/human_w_lexicon.csv',
      #                           non_word_task=folder_in + 'temp_files/nw_file_empty.txt')

    #wrapper_lexicon_wide_pe_estimation(human_lex, file_name=folder_out+"preview_0_bsa_6.csv", ope_version="gagl_2020", word_in_lex=True)

   ##################### PREVIEW oPE/difference measurement
    preview = 3
    word_prepro = read_csv(folder_in + "wordlist_for_Benjamin.txt")
    out = [return_df_preview(stims=word_prepro, stim=stim, folder_in=folder_in, preview=preview, preview_diff=True, ope_version="fu_2024", threshold=0.1) for
           stim in word_prepro["string"]]
    pd.concat(out).to_csv(folder_out + "preview_difference_" + str(preview) + ".csv", index=False)

    #out = return_df_preview(stims=word_prepro, stim="Abtei", folder_in=folder_in, preview=preview)
    #print(out)



#non_word = "pw"
#dataset = "train"
#lex_N_words = "500"
#    print("### Generate oPE Images ###")
# non_words = ["pw", "cs"]
# datasets = ["train", "test"]
# lex_N_words = ["500", "1800"]

#    for non_word in non_words:
#       for dataset in datasets:
#          for lex_N_word in lex_N_words:
#             path = "/Users/bg/PycharmProjects/py_slr_v.01/images/task_w_" + non_word + "/orig_oPE_lex_" + lex_N_word + "/" + dataset + "/"
#            print("### Generate - oPE images "+ non_word + " .. " + lex_N_word + " .. " + dataset +" ###")
#           human_lex = initiate_lexicon(word_file_task="./images/lexicon_task_lists/fu_oPE_" + dataset + "_w.txt",
#                                       word_file_lexicon="./images/lexicon_task_lists/lex_" + lex_N_word + ".txt",
#                                      non_word_task="./images/lexicon_task_lists/fu_oPE_" + dataset + "_" + non_word + ".txt")
#
#               print(path)
#              wrapper_store_oPE_perc(lexicon=human_lex,
#                                    threshold=.5,
#                                   mode="mean",
#                                  path=path,
#                                 height_image=64,
#                                width_image=64,
#                               font_path="fonts/cmuntt.ttf"
#                              )

#    print("### Simulate - learning data ###")
#   folder_in = "/Users/bg/PycharmProjects/SLR_Model/stimuli_human/pwv_learned/"
#  folder_out = "/Users/bg/PycharmProjects/Py_SLR_v0.01/simulation_output_files/"
# animals = ["BA0806", "BN2211", "CG2608", "CM2901", "DA1403", "GE2905", "GN0605", "GN1201", "GR0809", "HA0407",
#           "PS0807", "PS2510", "RA0411", "RE2603", "RN0412", "KE1307", "KE1501", "KH0202", "KN0611", "LA0502",
#          "LL1309", "HA2212", "HN2105", "HS0501", "MA2209", "PE2208", "MA2909", "MA3105", "NL0506", "PA0709",
#         "SE0609", "SN1003", "SS2505", "TA1511", "UA1501", "WA0210", "WL2810"]

#    sessions = [1, 2, 3, 4]
#   for animal in animals:
#      for session in sessions:
#         presented = read_csv(folder_in + animal + "_presented_pws.csv")
#        learned = read_csv(folder_in + animal + "_learned_pwv.csv")

#       presented[(presented["session"] == session) & (presented["condition"] == "pwv")]["pw"].to_csv(
#          folder_out + "w_task_tmp.csv", header=False, index=False)
#     presented[(presented["session"] == session) & (presented["condition"] == "fill")]["pw"].to_csv(
#        folder_out + "nw_task_tmp.csv", header=False, index=False)
# learned[learned["session"] == session]["pwl"].to_csv(folder_out + "lex_tmp.csv", header=False, index=False)

#   human_lex = initiate_lexicon(word_file_task=folder_out + 'w_task_tmp.csv',
#                               word_file_lexicon=folder_out + 'nw_task_tmp.csv',
#                              non_word_task=folder_out + 'nw_task_tmp.csv')

# wrapper_lexicon_wide_pe_estimation(human_lex,
#                                  file_name=folder_out + "human_pw_learn_data/" + animal + "_" + str(
#                                     session) + "_no_overlap_in_lex.csv", word_in_lex=True)

#    print("### Simulate - frequency dependent PEs ###")
#   lex_5L = pd.read_csv(
#      "/Users/bg/sciebo/vOT_model/papers/ope/Science_submission_data/oPE_func_data_analysis_pub/ger5_fin.csv")
# data_lexdec = pd.read_csv(
#    "/Users/bg/sciebo/vOT_model/papers/ope/Science_submission_data/oPE_func_data_analysis_pub/lex_dec.csv")
#
#   folder_tmp_freq_list = "/Users/bg/PycharmProjects/Py_SLR_v0.01/simulation_output_files/Freq_lex_datasets/"
#  w_task = data_lexdec["string"].loc[(data_lexdec["vp"] == "AB234") & (data_lexdec["category"] == "Word")]
# nw_task = data_lexdec["string"].loc[(data_lexdec["vp"] == "AB234") & (data_lexdec["category"] != "Word")]
# with open(folder_tmp_freq_list + "w_task_tmp.csv", 'w') as f:
#   f.write('\n'.join(map(str, w_task.tolist())))
#    with open(folder_tmp_freq_list + "nw_task_tmp.csv", 'w') as f:
#       f.write('\n'.join(map(str, nw_task.tolist())))
#
#   n_words = list(range(100, 2001, 100))
#  random_samples = list(range(1, 11, 1))

#    for n in n_words:
#       lex = lex_5L["strings"].iloc[:n]
#      #lex.to_csv(folder_tmp_freq_list + "lex_tmp.csv")
#     with open(folder_tmp_freq_list + "lex_tmp.csv", 'w') as f:
#        f.write('\n'.join(map(str, lex.tolist())))
#
#       human_lex = initiate_lexicon(word_file_task=folder_tmp_freq_list + 'w_task_tmp.csv',
#                                   word_file_lexicon=folder_tmp_freq_list + 'lex_tmp.csv',
#                                  non_word_task=folder_tmp_freq_list + 'nw_task_tmp.csv')
#    wrapper_lexicon_wide_pe_estimation(human_lex,
#                                      file_name=folder_tmp_freq_list + "Most_frequent_N_" + str(
#                                         n) + "_words_in_lex.csv", word_in_lex=True)
#
#       for sample in random_samples:
#          lex = lex_5L["strings"].sample(n=n, random_state=sample)
#         with open(folder_tmp_freq_list + "lex_tmp.csv", 'w') as f:
#            f.write('\n'.join(map(str, lex.tolist())))
#
#           human_lex = initiate_lexicon(word_file_task=folder_tmp_freq_list + 'w_task_tmp.csv',
#                                       word_file_lexicon=folder_tmp_freq_list + 'lex_tmp.csv',
#                                      non_word_task=folder_tmp_freq_list + 'nw_task_tmp.csv')
#        wrapper_lexicon_wide_pe_estimation(human_lex,
#                                          file_name=folder_tmp_freq_list + "Random_N_" + str(
#                                             n) +"_"+ str(sample) +"_words_in_lex.csv", word_in_lex=True)
#


#   # --------------------------------------------------------------------------------------------------------------
#  # -------------------------------------------50p_oPE------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------
#    path = "/Users/bg/PycharmProjects/py_slr_v.01/images/task_w_pw/50p_oPE/train/words/"
#   print(path)
#  wrapper_store_oPE_perc(lexicon=folder_in + 'human_w_lexicon.csv',
#                        word_list=word_list_train,  # [str(example_word_list[i]) for i in my_list_train],
#                       font_path=font_path,
#                      height_image=height_image,
#                     width_image=width_image,
#                    threshold=.5,
#                   mode="binary",
#                  path=path)
#
#   path = path_output + "task_w_pw/50p_oPE/test/words/"
#  print(path)
# wrapper_store_oPE_perc(lexicon=lexicon_l5_ger,
#                       word_list=word_list_test,  # [str(example_word_list[i]) for i in my_list_train],
#                      font_path=font_path,
#                     height_image=height_image,
#                    width_image=width_image,
#                   threshold=.5,
#                  mode="mean",
#                 path=path)
#
#   path = path_output + "task_w_pw/50p_oPE/train/non-words/"
#  print(path)
# wrapper_store_oPE_perc(lexicon=lexicon_l5_ger,
#                       word_list=pw_list_train,  # [str(example_word_list[i]) for i in my_list_train],
#                      font_path=font_path,
#                     height_image=height_image,
#                    width_image=width_image,
#                   threshold=.5,
#                  mode="mean",
#                 path=path)
#
#   path = path_output + "task_w_pw/50p_oPE/test/non-words/"
#  print(path)
# wrapper_store_oPE_perc(lexicon=lexicon_l5_ger,
#                       word_list=pw_list_test,  # [str(example_word_list[i]) for i in my_list_train],
#                      font_path=font_path,
#                     height_image=height_image,
#                    width_image=width_image,
#                   threshold=.5,
#                  mode="mean",
#                 path=path)
# -----#---------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
#    path = path_output + "task_w_cs/50p_oPE/train/words/"
#   print(path)
#  wrapper_store_oPE_perc(lexicon=lexicon_l5_ger,
#                        word_list=word_list_train,  # [str(example_word_list[i]) for i in my_list_train],
#                       font_path=font_path,
#                      height_image=height_image,
#                     width_image=width_image,
#                    threshold=.5,
#                   mode="mean",
#                  path=path)
#
#   path = path_output + "task_w_cs/50p_oPE/test/words/"
#  print(path)
# wrapper_store_oPE_perc(lexicon=lexicon_l5_ger,
#                       word_list=word_list_test,  # [str(example_word_list[i]) for i in my_list_train],
#                      font_path=font_path,
#                     height_image=height_image,
#                    width_image=width_image,
#                   threshold=.5,
#                  mode="mean",
#                 path=path)
#
#   path = path_output + "task_w_cs/50p_oPE/train/non-words/"
#  print(path)
# wrapper_store_oPE_perc(lexicon=lexicon_l5_ger,
#                       word_list=cs_list_train,  # [str(example_word_list[i]) for i in my_list_train],
#                      font_path=font_path,
#                     height_image=height_image,
#                    width_image=width_image,
#                   threshold=.5,
#                  mode="mean",
#                 path=path)
#
#   path = path_output + "task_w_cs/50p_oPE/test/non-words/"
#  print(path)
# wrapper_store_oPE_perc(lexicon=lexicon_l5_ger,
#                       word_list=cs_list_test,  # [str(example_word_list[i]) for i in my_list_train],
#                      font_path=font_path,
#                     height_image=height_image,
#                    width_image=width_image,
#                   threshold=.5,
#                  mode="mean",
#                 path=path)
#
