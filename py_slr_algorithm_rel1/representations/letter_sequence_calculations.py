from collections import Counter
from numpy import sum


def estimate_letter_sequence_probabilities(lexicon, invert_str=False, stimtype_lexicon=["lexicon_words"]):
    for stimtype in stimtype_lexicon:
        constant = len(lexicon[stimtype])
        if constant !=0:
            min_n_letters = max([len(string) for string in lexicon[stimtype]])
            for i in range(1, min_n_letters):
                if invert_str:
                    pos_i_letters = [string[::-1][0:i] for string in lexicon[stimtype] if i + 1 <= len(string)]
                    lexicon[stimtype + "_pos_" + str(i) + "_letter_sequence_freq_abs_inv"] = Counter(pos_i_letters)
                    lexicon[stimtype + "_pos_" + str(i) + "_letter_sequence_freq_inv"] = {key: value / constant for
                                                                                          key, value
                                                                                          in
                                                                                          lexicon[stimtype + "_pos_" + str(
                                                                                              i) + "_letter_sequence_freq_abs_inv"].items()}

                else:
                    pos_i_letters = [string[0:i] for string in lexicon[stimtype] if i + 1 <= len(string)]
                    lexicon[stimtype + "_pos_" + str(i) + "_letter_sequence_freq_abs"] = Counter(pos_i_letters)
                    lexicon[stimtype + "_pos_" + str(i) + "_letter_sequence_freq"] = {key: value / constant for key, value
                                                                                      in lexicon[stimtype + "_pos_" + str(
                            i) + "_letter_sequence_freq_abs"].items()}


def get_word_spe(string, lexicon, output="value", threshold="No"):
    pe = 0
    pe_list = []
    for i in range(1, len(string)):
        if string[0:i] in lexicon["lexicon_words_pos_" + str(i) + "_letter_sequence_freq"]:
            pe = (1 - lexicon["lexicon_words_pos_" + str(i) + "_letter_sequence_freq"][string[0:i]])
            if threshold == "No":
                pe_list.append(pe)
            else:
                if pe >= threshold:
                    pe_list.append(1)
                elif pe < threshold:
                    pe_list.append(0)
        else:
            pe_list.append(1)
    if output == "value":
        return sum(pe_list)
    elif output == "list":
        return pe_list


def get_word_spe_inverted(string, lexicon):
    pe = 0
    pe_list = []
    for i in range(1, len(string)):
        if string[0:i] in lexicon["lexicon_words_pos_" + str(i) + "_letter_sequence_freq_inv"]:
            pe = (1 - lexicon["lexicon_words_pos_" + str(i) + "_letter_sequence_freq_inv"][string[0:i]])
            pe_list.append(pe)
        else:
            pe_list.append(1)
    return sum(pe_list)
