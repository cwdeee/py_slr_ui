from typing import Literal

import scipy.stats as stats
from numpy import arange, std, mean, array

from representations.letter_position_calculations import \
    get_word_ape  # calculate_pos_letter_freq, calculate_pos_freq_differences, \
from representations.letter_sequence_calculations import \
    get_word_spe  # estimate_letter_sequence_probabilities, ,     get_word_spe_inverted
from representations.oPE_calculations import get_prediction_img, wrapper_multi_parameter_modelling_oPE

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression


class lexical_categorization_info:
    type: Literal["ope", "ape", "spe", "ope:ape", "ope:spe", "ape:spe", "ope:ape:spe"]
    lexicon: []
    min_pe: [int]
    max_pe: [int]
    pe_values: []
    percentiles: []

    def __init__(self, type="ope"):
        self.type = type


def find_decision_boundary(lexicon, type="ope"):
    len_words = array([len(string) for string in lexicon["task_words"]])

    ape_words = array([get_word_ape(string, lexicon) for string in lexicon["task_words"]])
    ape_words = ape_words / len_words
    spe_words = array([get_word_spe(string, lexicon, output="value") for string in lexicon["task_words"]]) / (
            len_words - 1)

    prediction_img = get_prediction_img(lexicon["lexicon_words"], noise_amount=0, threshold=0.5, mode="mean")
    ope_words = array(
        [wrapper_multi_parameter_modelling_oPE(string, prediction_img) for string in lexicon["task_words"]])
    ope_words = ope_words / 255

    if type == "ope":
        ope_lex_cat = define_lex_cat_class(lexicon, type="ope", pe_list=ope_words)
        return ope_lex_cat

    elif type == "ape":
        ape_lex_cat = define_lex_cat_class(lexicon, type="ape", pe_list=ape_words)
        return ape_lex_cat

    elif type == "spe":
        ape_lex_cat = define_lex_cat_class(lexicon, type="spe", pe_list=spe_words)
        return ape_lex_cat

    elif type == "ape:spe":
        pe_list = ape_words + spe_words
        ape_lex_cat = define_lex_cat_class(lexicon, type="ape:spe", pe_list=pe_list)
        return ape_lex_cat

    elif type == "ope:spe":
        pe_list = ope_words + spe_words
        ape_lex_cat = define_lex_cat_class(lexicon, type="ope:spe", pe_list=pe_list)
        return ape_lex_cat

    elif type == "ope:ape":
        pe_list = ape_words + ope_words
        ape_lex_cat = define_lex_cat_class(lexicon, type="ope:ape", pe_list=pe_list)
        return ape_lex_cat

    elif type == "ope:ape:spe":
        pe_list = ape_words + ope_words + spe_words
        ape_lex_cat = define_lex_cat_class(lexicon, type="ope:ape:spe", pe_list=pe_list)
        return ape_lex_cat


def define_lex_cat_class(lexicon, type="ope", pe_list=[]):
    lex_cat = lexical_categorization_info
    lex_cat.type = type
    lex_cat.lexicon = lexicon
    lex_cat.min_pe = min(pe_list)
    lex_cat.max_pe = max(pe_list)
    lex_cat.pe_values = pe_list
    lex_cat.percentiles = stats.norm.ppf(arange(0.01, 1, 0.01), loc=mean(lex_cat.pe_values),
                                         scale=std(lex_cat.pe_values))
    return lex_cat


def lex_cat_decision_boundary(w_pe, nw_pe, kernel_size=0.5):
    # Calculate densities
    x_w_pe = gaussian_kde(w_pe, bw_method=kernel_size)
    x_nw_pe = gaussian_kde(nw_pe, bw_method=kernel_size)
    x = np.linspace(min(min(w_pe), min(nw_pe)), max(max(w_pe), max(nw_pe)), 1000)
    density_w = x_w_pe.evaluate(x)
    density_nw = x_nw_pe.evaluate(x)

    # Calculate probabilities
    pW = density_w / (density_w + density_nw)
    pNW = density_nw / (density_w + density_nw)

    # Calculate entropy
    entro = -pW * np.log2(pW) - pNW * np.log2(pNW)

    # Fit loess model
    loess_model = LinearRegression()
    loess_model.fit(x.reshape(-1, 1), entro)

    # Predict entropy
    predicted_entro = loess_model.predict(x.reshape(-1, 1))

    # Filter data within one standard deviation of the mean
    mean_val = np.mean(np.concatenate([w_pe, nw_pe]))
    std_val = np.std(np.concatenate([w_pe, nw_pe]))
    catentro = np.column_stack((x, density_w, density_nw, pW, pNW, entro, predicted_entro))
    catentro = catentro[(catentro[:, 0] > mean_val - std_val) & (catentro[:, 0] < mean_val + std_val)]

    # Find maximum predicted entropy
    max_entro_idx = np.argmax(catentro[:, -1][~np.isnan(catentro[:, -1])])
    decision_boundary = catentro[:, 0][~np.isnan(catentro[:, -1])][max_entro_idx]

    return decision_boundary


def lex_cat_entro(w_pe, nw_pe, string_pe, kernel_size=0.5):
    # Calculate densities
    x_w_pe = gaussian_kde(w_pe, bw_method=kernel_size)
    x_nw_pe = gaussian_kde(nw_pe, bw_method=kernel_size)
    x = np.linspace(min(min(w_pe), min(nw_pe)), max(max(w_pe), max(nw_pe)), 1000)
    density_w = x_w_pe.evaluate(x)
    density_nw = x_nw_pe.evaluate(x)

    # Calculate probabilities
    pW = density_w / (density_w + density_nw)
    pNW = density_nw / (density_w + density_nw)

    # Calculate entropy
    entro = -pW * np.log2(pW) - pNW * np.log2(pNW)

    return np.mean(entro[np.round(x, 3) == np.round(string_pe, 3)])
