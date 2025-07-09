from collections import Counter
from numpy import mean, std


def calculate_pos_letter_freq(lexicon, stimtype_lexicon=["lexicon_words"]):
    for stimtype in stimtype_lexicon:
        constant = len(lexicon[stimtype])
        if constant !=0:
            min_n_letters = max([len(string) for string in lexicon[stimtype]])
            for i in range(0, min_n_letters):
                pos_i_letters = [string[i] for string in lexicon[stimtype] if i + 1 <= len(string)]
                lexicon[stimtype + "_pos_" + str(i + 1) + "_letter_freq_abs"] = Counter(pos_i_letters)
                lexicon[stimtype + "_pos_" + str(i + 1) + "_letter_freq"] = {key: value / constant for key, value in
                                                                             lexicon[stimtype + "_pos_" + str(
                                                                                 i + 1) + "_letter_freq_abs"].items()}


def calculate_pos_freq_differences(lexicon):
    min_n_letters = min([len(string) for string in lexicon["task_words"]])
    for i in range(0, min_n_letters):
        dict_words = lexicon["words_pos_" + str(i + 1) + "_letter_freq"]
        dict_nonwords = lexicon["non-words_pos_" + str(i + 1) + "_letter_freq"]
        shared_letters = [key for key, value in dict_words.items() if key in dict_nonwords and dict_words]
        lexicon["W_NW_ABS_difference_pos_" + str(i + 1) + "_letter_freq"] = {
            letter: abs(dict_nonwords[letter] - dict_words[letter]) for letter in shared_letters}

        values = list(lexicon["W_NW_ABS_difference_pos_" + str(i + 1) + "_letter_freq"].values())
        print("Letter position " + str(i + 1) + " mean:", mean(values))
        print("Letter position " + str(i + 1) + " SD:", std(values))


def get_word_ape(string, lexicon, output="value", threshold="No"):
    pe = 0
    pe_list = []
    for i in range(0, len(string)):
        if string[i] in lexicon["lexicon_words_pos_" + str(i + 1) + "_letter_freq"]:
            pe = (1 - lexicon["lexicon_words_pos_" + str(i + 1) + "_letter_freq"][string[i]])
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
