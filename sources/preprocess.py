import re
from nltk.tokenize import word_tokenize
import pandas as pd
import stop_words
import unicodedata
import spacy
from typing import List
import ujson
import os
from itertools import groupby
import operator
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sources.utils import json_file_dict


class Preprocess:
    def __init__(self, text_serie: pd.core.series.Series):
        self.series = text_serie
        self.stopwords = Preprocess._stopwords_list()

    @classmethod
    def _stopwords_list(cls):
        """
        Create a personnalized list of stopwords
        """
        general_stopwords = stop_words.get_stop_words(language="fr")
        personnalized_stopwords = [
            "lundi",
            "mardi",
            "mercredi",
            "jeudi",
            "vendredi",
            "samedi",
            "dimanche",
        ]
        try:
            general_stopwords.remove("mot")
        except:
            print("Can't remove mot from stopword list")
        return general_stopwords + personnalized_stopwords

    @classmethod
    def clean_simple(cls, text_series: pd.core.series.Series, stopwords: List):
        """
        Clean each text :
        - puts each word in lower case,
        - does not keep words belonging to the stopwords list,
        - does not keep words with less than 2 characters,
        - keeps only letters and deletes numbers and special characters.
        """
        # create a list of words
        text_series = text_series.map(lambda text: word_tokenize(text))

        # puts each word in lower case, does not keep words belonging to the stopwords list, does not keep words with less than 2 characters,
        # keeps only letters and deletes numbers and special characters.
        text_series = text_series.map(
            lambda tok: [
                word.lower()
                for word in re.split(" ", re.sub(r"(\W+|_|\d+)", " ", " ".join(tok)))
                if word.lower() not in stopwords and len(word) > 1
            ]
        )
        return text_series

    def subsitution(self, lookup_dict: dict):
        """
        """
        self.series = self.series.apply(
            Preprocess.__subsititution_text, args=(lookup_dict,)
        )

    def clean_character(self):
        """
        """
        self.series = self.series.map(lambda text: re.sub(r"(\W+|_|\d+)", " ", text))
        self.series = self.series.map(lambda text: re.sub(r" +", " ", text))

    def clean_lemmatization(self, exceptions: dict):
        """
        """
        stopwords = self.stopwords
        nlp = spacy.load("fr_core_news_md")
        self.series = self.series.map(lambda text: nlp(text))
        self.series = self.series.apply(
            Preprocess.__lemmatization, args=(exceptions, stopwords,)
        )

    def display_unknow_word(
        self,
        related_df: pd.core.frame.DataFrame,
        path_data_ref: str,
        costum_known_words: List = [],
    ):
        """
        """
        dict_unknown = Preprocess.__unknow_word(
            self.series, costum_known_words, path_data_ref
        )
        for key, values in dict_unknown.items():
            print("--------")
            print(f"Element: {key}")
            print(f"Sentence: {related_df['commentaire_brut'].iloc[key]}")
            print(f"unkown words: {values}")
        return dict_unknown

    def plot_top_occur(self, N):
        new_list = [item for text in self.series for item in text.split()]
        freq = {key: len(list(group)) for key, group in groupby(np.sort(new_list))}
        sorted_freq = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)

        tags_N = []
        freq_N = []
        for i in range(N):
            tags_N.append(sorted_freq[i][0])
            freq_N.append(sorted_freq[i][1])

        height = N * 0.4

        fig = plt.figure(figsize=(18, height))
        sns.barplot(y=tags_N, x=freq_N)
        plt.title(
            "Number of occurrences of words (Top" + str(N) + str(")"), fontsize=18
        )
        plt.show()

    @classmethod
    def __subsititution_text(cls, input_text, lookup_dict: dict):
        if isinstance(input_text, str):
            for word in list(lookup_dict.keys()):
                new_text = re.sub(word, lookup_dict[word], input_text)
        if isinstance(input_text, list):
            words = input_text.copy()
            new_words = []
            for word in words:
                if word.lower() in lookup_dict:
                    word = lookup_dict[word.lower()]
                new_words.append(word)
            new_text = " ".join(new_words)
        return new_text

    @classmethod
    def __lemmatization(cls, doc, exceptions: List, stopwords: List):
        result_list = []
        for token in doc:
            if token.text.lower() in exceptions:
                result_list.append(token.text.lower())
            elif token.text.lower() not in stopwords and len(token.text) > 1:
                result_list.append(token.lemma_.lower())
        new_text = " ".join(result_list)
        return new_text

    @classmethod
    def __unknow_word(
        cls,
        text_series: pd.core.series.Series,
        costum_known_words: List,
        path_data_ref: str,
    ):
        """
        """
        file_path_1 = os.path.join(path_data_ref, "lookup_french_word.json")
        dict_french = json_file_dict(file_path_1)

        file_path_2 = os.path.join(path_data_ref, "index_french.json")
        dict_french_index = json_file_dict(file_path_2)

        known_words = (
            dict_french_index["adj"]
            + dict_french_index["adv"]
            + dict_french_index["noun"]
            + dict_french_index["verb"]
            + costum_known_words
        )

        dict_unknown = dict()
        for compt in range(text_series.size):
            words_list = text_series[compt].split()
            for word in words_list:
                if word not in dict_french:
                    if word not in known_words:
                        if compt not in dict_unknown:
                            dict_unknown[compt] = [word]
                        else:
                            dict_unknown[compt].append(word)
        return dict_unknown
