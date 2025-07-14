from behavior.walker import random_walk_with_boundary
from representations.letter_position_calculations import calculate_pos_letter_freq, \
    get_word_ape
from representations.letter_sequence_calculations import estimate_letter_sequence_probabilities, get_word_spe, \
    get_word_spe_inverted
from representations.oPE_calculations import get_prediction_img, wrapper_multi_parameter_modelling_oPE, show_image, \
    get_prediction_error_array
from behavior.lexical_categorization import find_decision_boundary, lex_cat_decision_boundary, lex_cat_entro
from pandas import DataFrame, read_csv
from numpy import arange, round, array
from random import choice
from PIL import Image


def import_file(file_path="file"):
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
        loaded_list = [line for line in lines]

        return loaded_list


def min_max_scaling(numbers):
    min_val = min(numbers)
    max_val = max(numbers)
    scaled_numbers = [(x - min_val) / (max_val - min_val) for x in numbers]
    return scaled_numbers


def initiate_lexicon(word_file_task, word_file_lexicon, non_word_task):
    tmp_w = read_csv(word_file_task)
    tmp_nw = read_csv(non_word_task)

    lexicon = {
        "task_words": tmp_w[tmp_w.columns.tolist()[0]].to_list() + tmp_w.columns.tolist(),
        "lexicon_words": import_file(word_file_lexicon),
        "task_non-words": tmp_nw[tmp_nw.columns.tolist()[0]].to_list() + tmp_nw.columns.tolist()
    }
    calculate_pos_letter_freq(lexicon)
    estimate_letter_sequence_probabilities(lexicon)
    estimate_letter_sequence_probabilities(lexicon, invert_str=True)
    return lexicon


def lex_cat(representation="ope_norm", boundary=50, df=[], lexicon=[]):
    if boundary == "LCM":
        boundary = lex_cat_decision_boundary(w_pe=df[representation][df["lexicality"] == "word"],
                                             nw_pe=df[representation][df["lexicality"] == "non-word"])
        df[representation + "_dec_LCM_" + str(round(boundary, decimals=2))] = 0
        df.loc[((df[representation]) > boundary) & (df["lexicality"] == "non-word"), representation + "_dec_LCM_" + str(
            round(boundary, decimals=2))] = 1
        df.loc[((df[representation]) <= boundary) & (df["lexicality"] == "word"), representation + "_dec_LCM_" + str(
            round(boundary, decimals=2))] = 1

    else:
        df[representation + "_dec_p" + str(boundary)] = 0
        df.loc[((df[representation]) > find_decision_boundary(lexicon).percentiles[boundary]) & (
                df["lexicality"] == "non-word"), representation + "_dec_p" + str(boundary)] = 1
        df.loc[((df[representation]) <= find_decision_boundary(lexicon).percentiles[boundary]) & (
                df["lexicality"] == "word"), representation + "_dec_p" + str(boundary)] = 1


def wrapper_lexicon_wide_pe_estimation(lexicon, file_name="tmp.csv", boundaries=arange(60, 70, 30), word_in_lex=False,
                                       rt_sim=False, dec_boundary=False, ape_threshold="No", spe_threshold="No",
                                       ope_mode="mean", ope_version="fu_2024"):
    if word_in_lex == False:
        data = {'string': list(lexicon["task_words"]) + list(lexicon["task_non-words"]),
                'lexicality': ["word"] * len(lexicon["task_words"]) + ["non-word"] * len(lexicon["task_non-words"]),
                'ope': 0,
                'ape': 0,
                'spe': 0,
               # 'spe_inverted': 0,
                'letter_length': 0,
                "ope_norm": 0,
                "ape_norm": 0,
                "spe_norm": 0,
                'in_lexicon': False
                }
        df = DataFrame(data)

        for i in boundaries:
            df["ope_norm_dec_p" + str(i)] = [choice([0, 1]) for word in df["string"]]
            df["ape_norm_dec_p" + str(i)] = [choice([0, 1]) for word in df["string"]]
            df["spe_norm_dec_p" + str(i)] = [choice([0, 1]) for word in df["string"]]
            df["aspe_norm_dec_p" + str(i)] = [choice([0, 1]) for word in df["string"]]
            df["ospe_norm_dec_p" + str(i)] = [choice([0, 1]) for word in df["string"]]
            df["oape_norm_dec_p" + str(i)] = [choice([0, 1]) for word in df["string"]]
            df["oaspe_norm_dec_p" + str(i)] = [choice([0, 1]) for word in df["string"]]

        print(file_name)
        df.to_csv(file_name)
    else:
        ape_words = [get_word_ape(string, lexicon, output="value", threshold=ape_threshold) for string in
                     lexicon["task_words"]]
        spe_words = [get_word_spe(string, lexicon, output="value", threshold=spe_threshold) for string in
                     lexicon["task_words"]]
       # spe_words_inv = [get_word_spe_inverted(string, lexicon) for string in lexicon["task_words"]]

        ape_N_words = [get_word_ape(string, lexicon, output="value", threshold=ape_threshold) for string in
                       lexicon["task_non-words"]]
        spe_N_words = [get_word_spe(string, lexicon, output="value", threshold=spe_threshold) for string in
                       lexicon["task_non-words"]]
        spe_N_words_inv = [get_word_spe_inverted(string, lexicon) for string in lexicon["task_non-words"]]

        prediction_img = get_prediction_img(lexicon["lexicon_words"], noise_amount=0, threshold=0.5, mode=ope_mode,
                                            version=ope_version)
        #show_image(prediction_img.prediction)
        ope_words = [wrapper_multi_parameter_modelling_oPE(string, prediction_img) for string in lexicon["task_words"]]
        ope_N_words = [wrapper_multi_parameter_modelling_oPE(string, prediction_img) for string in
                       lexicon["task_non-words"]]

        len_words = [len(string) for string in lexicon["task_words"]]
        len_N_words = [len(string) for string in lexicon["task_non-words"]]

        data = {'string': list(lexicon["task_words"]) + list(lexicon["task_non-words"]),
                'lexicality': ["word"] * len(lexicon["task_words"]) + ["non-word"] * len(lexicon["task_non-words"]),
                'ope': ope_words + ope_N_words,
                'ape': ape_words + ape_N_words,
                'spe': spe_words + spe_N_words,
             #   'spe_inverted': spe_words_inv + spe_N_words_inv,
                'letter_length': len_words + len_N_words}

        df = DataFrame(data)
        df['in_lexicon'] = df['string'].apply(lambda x: any(s in x for s in lexicon["lexicon_words"]))

        df["ope_norm"] = df["ope"] / (prediction_img.height_image * prediction_img.width_image)
        df["ape_norm"] = df["ape"] / df["letter_length"]
        df["spe_norm"] = df["spe"] / (df["letter_length"] - 1)
        df["oape_norm_sum"] = df["ope_norm"] + df["ape_norm"]
        df["ospe_norm_sum"] = df["ope_norm"] + df["spe_norm"]
        df["aspe_norm_sum"] = df["spe_norm"] + df["ape_norm"]
        df["oaspe_norm_sum"] = df["spe_norm"] + df["ope_norm"] + df["ape_norm"]
        # df["oaspe_norm_mult"] = df["spe_norm"] * df["ope_norm"] * df["ape_norm"]

        if dec_boundary:
            lex_cat(representation="ope_norm", boundary="LCM", df=df, lexicon=lexicon)
            lex_cat(representation="ape_norm", boundary="LCM", df=df, lexicon=lexicon)
            lex_cat(representation="spe_norm", boundary="LCM", df=df, lexicon=lexicon)

            lex_cat(representation="oaspe_norm_sum", boundary="LCM", df=df, lexicon=lexicon)
          #  lex_cat(representation="oaspe_norm_mult", boundary="LCM", df=df, lexicon=lexicon)

            df["ope_norm_entro"] = min_max_scaling([lex_cat_entro(w_pe=df["ope_norm"][df["lexicality"] == "word"],
                                                                  nw_pe=df["ope_norm"][df["lexicality"] == "non-word"],
                                                                  string_pe=pe) for pe in df["ope_norm"]])

            df["ape_norm_entro"] = min_max_scaling([lex_cat_entro(w_pe=df["ape_norm"][df["lexicality"] == "word"],
                                                                  nw_pe=df["ape_norm"][df["lexicality"] == "non-word"],
                                                                  string_pe=pe) for pe in df["ape_norm"]])

            df["spe_norm_entro"] = min_max_scaling([lex_cat_entro(w_pe=df["spe_norm"][df["lexicality"] == "word"],
                                                                  nw_pe=df["spe_norm"][df["lexicality"] == "non-word"],
                                                                  string_pe=pe) for pe in df["spe_norm"]])

            df["oaspe_norm_sum_entro"] = min_max_scaling(
                [lex_cat_entro(w_pe=df["oaspe_norm_sum"][df["lexicality"] == "word"],
                               nw_pe=df["oaspe_norm_sum"][df["lexicality"] == "non-word"],
                               string_pe=pe) for pe in df["oaspe_norm_sum"]])

           # df["oaspe_norm_mult_entro"] = min_max_scaling(
            #    [lex_cat_entro(w_pe=df["oaspe_norm_mult"][df["lexicality"] == "word"],
             #                  nw_pe=df["oaspe_norm_mult"][df["lexicality"] == "non-word"],
              #                 string_pe=pe) for pe in df["oaspe_norm_mult"]])

        if rt_sim:
            df["ope_norm_entro_sim_rt"] = [random_walk_with_boundary(lc_difficutly=(diff - .5)) for diff in
                                           df["ope_norm_entro"]]
            df["ape_norm_entro_sim_rt"] = [random_walk_with_boundary(lc_difficutly=(diff - .5)) for diff in
                                           df["ape_norm_entro"]]
            df["spe_norm_entro_sim_rt"] = [random_walk_with_boundary(lc_difficutly=(diff - .5)) for diff in
                                           df["spe_norm_entro"]]
            df["oaspe_norm_sum_entro_sim_rt"] = [random_walk_with_boundary(lc_difficutly=(diff - .5)) for diff in
                                                 df["oaspe_norm_sum_entro"]]
          #  df["oaspe_norm_mult_entro_sim_rt"] = [random_walk_with_boundary(lc_difficutly=(diff - .5)) for diff in
           #                                       df["oaspe_norm_mult_entro"]]
        if dec_boundary:
            for i in boundaries:
                lex_cat(representation="ope_norm", boundary=i, df=df, lexicon=lexicon)
                lex_cat(representation="ape_norm", boundary=i, df=df, lexicon=lexicon)
                lex_cat(representation="spe_norm", boundary=i, df=df, lexicon=lexicon)

                df["aspe_norm_dec_p" + str(i)] = 0
                df.loc[
                    ((df["ape_norm"] + df["spe_norm"]) > find_decision_boundary(lexicon, type="ape:spe").percentiles[
                        i]) & (
                            df["lexicality"] == "non-word"), "aspe_norm_dec_p" + str(i)] = 1
                df.loc[
                    ((df["ape_norm"] + df["spe_norm"]) <= find_decision_boundary(lexicon, type="ape:spe").percentiles[
                        i]) & (
                            df["lexicality"] == "word"), "aspe_norm_dec_p" + str(i)] = 1
                df["ospe_norm_dec_p" + str(i)] = 0
                df.loc[
                    ((df["ope_norm"] + df["spe_norm"]) > find_decision_boundary(lexicon, type="ope:spe").percentiles[
                        i]) & (
                            df["lexicality"] == "non-word"), "ospe_norm_dec_p" + str(i)] = 1
                df.loc[
                    ((df["ope_norm"] + df["spe_norm"]) <= find_decision_boundary(lexicon, type="ope:spe").percentiles[
                        i]) & (
                            df["lexicality"] == "word"), "ospe_norm_dec_p" + str(i)] = 1
                df["oape_norm_dec_p" + str(i)] = 0
                df.loc[
                    ((df["ope_norm"] + df["ape_norm"]) > find_decision_boundary(lexicon, type="ope:ape").percentiles[
                        i]) & (
                            df["lexicality"] == "non-word"), "oape_norm_dec_p" + str(i)] = 1
                df.loc[
                    ((df["ope_norm"] + df["ape_norm"]) <= find_decision_boundary(lexicon, type="ope:ape").percentiles[
                        i]) & (
                            df["lexicality"] == "word"), "oape_norm_dec_p" + str(i)] = 1
                df["oaspe_norm_dec_p" + str(i)] = 0
                df.loc[((df["ope_norm"] + df["ape_norm"] + df["spe_norm"]) >
                        find_decision_boundary(lexicon, type="ope:ape:spe").percentiles[i]) & (
                               df["lexicality"] == "non-word"), "oaspe_norm_dec_p" + str(i)] = 1
                df.loc[((df["ope_norm"] + df["ape_norm"] + df["spe_norm"]) <=
                        find_decision_boundary(lexicon, type="ope:ape:spe").percentiles[i]) & (
                               df["lexicality"] == "word"), "oaspe_norm_dec_p" + str(i)] = 1

        # print(file_name)
        df = df.rename(columns={'ape': 'Lpe', 'ape_norm': 'Lpe_norm', 'oape_norm_sum': 'oLpe_norm_sum', 'aspe_norm_sum': 'Lspe_norm_sum', 'oaspe_norm_sum': 'oLspe_norm_sum'})
        df.to_csv(file_name, index=False)


def wrapper_store_oPE_perc(lexicon, threshold=0.5, mode="mean", path="./images/", height_image=40, width_image=200,
                           font_path="fonts/Courier_Prime/CourierPrime-Bold.ttf"):
    print(f"IMPORTANT NOTE, this now will overwrite existing files in '{path}'. Continue? (y/n)")
    response = input().lower()
    if response == 'y':
        prediction_img = get_prediction_img(lexicon["lexicon_words"], noise_amount=0, threshold=threshold, mode=mode,
                                            height_image=height_image, width_image=width_image, font_path=font_path)
        #show_image(prediction_img.prediction)
        [save_image(get_prediction_error_array(word=str(word), prediction=prediction_img),
                    path=path + "words/" + str(word) + ".jpg") for word in lexicon["task_words"]]
        [save_image(get_prediction_error_array(word=str(word), prediction=prediction_img),
                    path=path + "non-words/" + str(word) + ".jpg") for word in lexicon["task_non-words"]]


def wrapper_lexicon_wide_pe_estimation_preview(lexicon, file_name="tmp.csv", boundaries=arange(60, 70, 30),
                                               word_in_lex=False,
                                               rt_sim=False, dec_boundary=False, ape_threshold="No", spe_threshold="No",
                                               ope_mode="mean", ope_version="fu_2024", preview=0, preview_diff=False,
                                               threshold=0.5):
    if word_in_lex == False:
        data = {'string': list(lexicon["task_words"]) + list(lexicon["task_non-words"]),
                'lexicality': ["word"] * len(lexicon["task_words"]) + ["non-word"] * len(lexicon["task_non-words"]),
                'ope': 0,
                'ape': 0,
                'spe': 0,
                'spe_inverted': 0,
                'letter_length': 0,
                "ope_norm": 0,
                "ape_norm": 0,
                "spe_norm": 0,
                'in_lexicon': False
                }
        df = DataFrame(data)
        if dec_boundary:
            for i in boundaries:
                df["ope_norm_dec_p" + str(i)] = [choice([0, 1]) for word in df["string"]]
                df["ape_norm_dec_p" + str(i)] = [choice([0, 1]) for word in df["string"]]
                df["spe_norm_dec_p" + str(i)] = [choice([0, 1]) for word in df["string"]]
                df["aspe_norm_dec_p" + str(i)] = [choice([0, 1]) for word in df["string"]]
                df["ospe_norm_dec_p" + str(i)] = [choice([0, 1]) for word in df["string"]]
                df["oape_norm_dec_p" + str(i)] = [choice([0, 1]) for word in df["string"]]
                df["oaspe_norm_dec_p" + str(i)] = [choice([0, 1]) for word in df["string"]]

        print(file_name)
        df.to_csv(file_name)
    else:
        #        ape_words = [get_word_ape(string, lexicon, output="value", threshold=ape_threshold) for string in
        #                    lexicon["task_words"]]
        #      spe_words = [get_word_spe(string, lexicon, output="value", threshold=spe_threshold) for string in
        #                  lexicon["task_words"]]
        #    spe_words_inv = [get_word_spe_inverted(string, lexicon) for string in lexicon["task_words"]]
        #
        #       ape_N_words = [get_word_ape(string, lexicon, output="value", threshold=ape_threshold) for string in
        #                     lexicon["task_non-words"]]
        #     spe_N_words = [get_word_spe(string, lexicon, output="value", threshold=spe_threshold) for string in
        #                   lexicon["task_non-words"]]
        #   spe_N_words_inv = [get_word_spe_inverted(string, lexicon) for string in lexicon["task_non-words"]]
        if preview_diff:
            stim = lexicon["task_words"][0]
            mask = "Xxxxxx"
            mask = mask[:len(stim)]

            preview_vision = [stim[:preview] + mask[(preview):]]
            prediction_img = get_prediction_img(preview_vision, noise_amount=0, threshold=threshold, mode=ope_mode,
                                                version=ope_version)
            #show_image(prediction_img.prediction)
        else:
            prediction_img = get_prediction_img(lexicon["lexicon_words"], noise_amount=0, threshold=threshold,
                                                mode=ope_mode,
                                                version=ope_version)
            #show_image(prediction_img.prediction)

        ope_words = [wrapper_multi_parameter_modelling_oPE(string, prediction_img) for string in lexicon["task_words"]]
        ope_N_words = [wrapper_multi_parameter_modelling_oPE(string, prediction_img) for string in
                       lexicon["task_non-words"]]

        len_words = [len(string) for string in lexicon["task_words"]]
        len_N_words = [len(string) for string in lexicon["task_non-words"]]

        data = {'string': list(lexicon["task_words"]) + list(lexicon["task_non-words"]),
                'lexicality': ["word"] * len(lexicon["task_words"]) + ["non-word"] * len(lexicon["task_non-words"]),
                'ope': ope_words + ope_N_words,
                #       'ape': ape_words + ape_N_words,
                #      'spe': spe_words + spe_N_words,
                #     'spe_inverted': spe_words_inv + spe_N_words_inv,
                'letter_length': len_words + len_N_words}

        df = DataFrame(data)
        df['preview'] = preview
        df['in_lexicon'] = df['string'].apply(lambda x: any(s in x for s in lexicon["lexicon_words"]))

        df["ope_norm"] = df["ope"] / (prediction_img.height_image * prediction_img.width_image)
        #  df["ape_norm"] = df["ape"] / df["letter_length"]
        # df["spe_norm"] = df["spe"] / (df["letter_length"] - 1)
        #df["oape_norm_sum"] = df["ope_norm"] + df["ape_norm"]
        #    df["ospe_norm_sum"] = df["ope_norm"] + df["spe_norm"]
        #   df["aspe_norm_sum"] = df["spe_norm"] + df["ape_norm"]
        #  df["oaspe_norm_sum"] = df["spe_norm"] + df["ope_norm"] + df["ape_norm"]
        # df["oaspe_norm_mult"] = df["spe_norm"] * df["ope_norm"] * df["ape_norm"]

        if dec_boundary:
            lex_cat(representation="ope_norm", boundary="LCM", df=df, lexicon=lexicon)
            lex_cat(representation="ape_norm", boundary="LCM", df=df, lexicon=lexicon)
            lex_cat(representation="spe_norm", boundary="LCM", df=df, lexicon=lexicon)

            lex_cat(representation="oaspe_norm_sum", boundary="LCM", df=df, lexicon=lexicon)
            lex_cat(representation="oaspe_norm_mult", boundary="LCM", df=df, lexicon=lexicon)

            df["ope_norm_entro"] = min_max_scaling([lex_cat_entro(w_pe=df["ope_norm"][df["lexicality"] == "word"],
                                                                  nw_pe=df["ope_norm"][df["lexicality"] == "non-word"],
                                                                  string_pe=pe) for pe in df["ope_norm"]])

            df["ape_norm_entro"] = min_max_scaling([lex_cat_entro(w_pe=df["ape_norm"][df["lexicality"] == "word"],
                                                                  nw_pe=df["ape_norm"][df["lexicality"] == "non-word"],
                                                                  string_pe=pe) for pe in df["ape_norm"]])

            df["spe_norm_entro"] = min_max_scaling([lex_cat_entro(w_pe=df["spe_norm"][df["lexicality"] == "word"],
                                                                  nw_pe=df["spe_norm"][df["lexicality"] == "non-word"],
                                                                  string_pe=pe) for pe in df["spe_norm"]])

            df["oaspe_norm_sum_entro"] = min_max_scaling(
                [lex_cat_entro(w_pe=df["oaspe_norm_sum"][df["lexicality"] == "word"],
                               nw_pe=df["oaspe_norm_sum"][df["lexicality"] == "non-word"],
                               string_pe=pe) for pe in df["oaspe_norm_sum"]])

            df["oaspe_norm_mult_entro"] = min_max_scaling(
                [lex_cat_entro(w_pe=df["oaspe_norm_mult"][df["lexicality"] == "word"],
                               nw_pe=df["oaspe_norm_mult"][df["lexicality"] == "non-word"],
                               string_pe=pe) for pe in df["oaspe_norm_mult"]])

        if rt_sim:
            df["ope_norm_entro_sim_rt"] = [random_walk_with_boundary(lc_difficutly=(diff - .5)) for diff in
                                           df["ope_norm_entro"]]
            df["ape_norm_entro_sim_rt"] = [random_walk_with_boundary(lc_difficutly=(diff - .5)) for diff in
                                           df["ape_norm_entro"]]
            df["spe_norm_entro_sim_rt"] = [random_walk_with_boundary(lc_difficutly=(diff - .5)) for diff in
                                           df["spe_norm_entro"]]
            df["oaspe_norm_sum_entro_sim_rt"] = [random_walk_with_boundary(lc_difficutly=(diff - .5)) for diff in
                                                 df["oaspe_norm_sum_entro"]]
            df["oaspe_norm_mult_entro_sim_rt"] = [random_walk_with_boundary(lc_difficutly=(diff - .5)) for diff in
                                                  df["oaspe_norm_mult_entro"]]
        if dec_boundary:
            for i in boundaries:
                lex_cat(representation="ope_norm", boundary=i, df=df, lexicon=lexicon)
                lex_cat(representation="ape_norm", boundary=i, df=df, lexicon=lexicon)
                lex_cat(representation="spe_norm", boundary=i, df=df, lexicon=lexicon)

                df["aspe_norm_dec_p" + str(i)] = 0
                df.loc[
                    ((df["ape_norm"] + df["spe_norm"]) > find_decision_boundary(lexicon, type="ape:spe").percentiles[
                        i]) & (
                            df["lexicality"] == "non-word"), "aspe_norm_dec_p" + str(i)] = 1
                df.loc[
                    ((df["ape_norm"] + df["spe_norm"]) <= find_decision_boundary(lexicon, type="ape:spe").percentiles[
                        i]) & (
                            df["lexicality"] == "word"), "aspe_norm_dec_p" + str(i)] = 1
                df["ospe_norm_dec_p" + str(i)] = 0
                df.loc[
                    ((df["ope_norm"] + df["spe_norm"]) > find_decision_boundary(lexicon, type="ope:spe").percentiles[
                        i]) & (
                            df["lexicality"] == "non-word"), "ospe_norm_dec_p" + str(i)] = 1
                df.loc[
                    ((df["ope_norm"] + df["spe_norm"]) <= find_decision_boundary(lexicon, type="ope:spe").percentiles[
                        i]) & (
                            df["lexicality"] == "word"), "ospe_norm_dec_p" + str(i)] = 1
                df["oape_norm_dec_p" + str(i)] = 0
                df.loc[
                    ((df["ope_norm"] + df["ape_norm"]) > find_decision_boundary(lexicon, type="ope:ape").percentiles[
                        i]) & (
                            df["lexicality"] == "non-word"), "oape_norm_dec_p" + str(i)] = 1
                df.loc[
                    ((df["ope_norm"] + df["ape_norm"]) <= find_decision_boundary(lexicon, type="ope:ape").percentiles[
                        i]) & (
                            df["lexicality"] == "word"), "oape_norm_dec_p" + str(i)] = 1
                df["oaspe_norm_dec_p" + str(i)] = 0
                df.loc[((df["ope_norm"] + df["ape_norm"] + df["spe_norm"]) >
                        find_decision_boundary(lexicon, type="ope:ape:spe").percentiles[i]) & (
                               df["lexicality"] == "non-word"), "oaspe_norm_dec_p" + str(i)] = 1
                df.loc[((df["ope_norm"] + df["ape_norm"] + df["spe_norm"]) <=
                        find_decision_boundary(lexicon, type="ope:ape:spe").percentiles[i]) & (
                               df["lexicality"] == "word"), "oaspe_norm_dec_p" + str(i)] = 1

        #print(df)
        return (df)


def return_df_preview(stims, stim, folder_in, preview=0, ope_version="gagl_2020", preview_diff=False, threshold=0.5):
    stims.loc[stims['string'] == stim, ['string']].to_csv(
        folder_in + 'temp_files/human_w.csv',
        index=False, header=False)
    print(stim)
    bsa = len(stim)
    preview_letters = stim[:preview]
    lex_prepro = list(read_csv(folder_in + "lexicon_upper_case_all_lengths.csv")["word"])

    if stim not in lex_prepro:
        lex_prepro.append(stim)

    df = DataFrame({"word": lex_prepro, "bsa": [len(s) for s in lex_prepro]})

    df.loc[(df['bsa'] == bsa) & (df['word'].str.startswith(preview_letters)), ['word']].to_csv(
        folder_in + 'temp_files/human_w_lexicon.csv', index=False,
        header=False)  # lex_prepro.loc[lex_prepro['bsa'] == bsa, ['word']]

    human_lex = initiate_lexicon(word_file_task=folder_in + 'temp_files/human_w.csv',
                                 word_file_lexicon=folder_in + 'temp_files/human_w_lexicon.csv',
                                 non_word_task=folder_in + 'temp_files/nw_file_empty.txt')

    out = wrapper_lexicon_wide_pe_estimation_preview(human_lex, ope_version=ope_version, word_in_lex=True,
                                                     preview=preview, preview_diff=preview_diff, threshold=threshold)
    return out[out['lexicality'] == 'word']


def save_image(word_image_array, path):
    img = Image.fromarray(word_image_array.pe)
    img = img.convert('L')
    return img.save(path)
