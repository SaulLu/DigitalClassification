import pandas as pd
import numpy as np
import os

from sources.model import Model
from sources.preprocess import Preprocess
from sources.utils import json_file_dict


def test_model(model_instance, classifier, vectorizer, raw_comment):
    if isinstance(raw_comment, str):
        data = np.array([raw_comment])
    if isinstance(raw_comment, np.ndarray):
        data = raw_comment
    series = pd.Series(data)
    com_preprocess = Preprocess(series)

    json_sub_path = os.path.join(os.environ["Data_Ref"], "custom_lookup_table.json")
    dict_sub = json_file_dict(json_sub_path)
    com_preprocess.subsitution(dict_sub)

    exception_path = os.path.join(
        os.environ["Data_Ref"], "exceptions_lemmatization.json"
    )
    dict_exceptions = json_file_dict(exception_path)
    list_exceptions = dict_exceptions["exceptions"].copy()

    com_preprocess.clean_character()
    com_preprocess.clean_lemmatization(list_exceptions)

    vect = model_instance.vectorizer[vectorizer]
    com_vectorized = vect.transform(com_preprocess.series.fillna("erreur"))
    model = model_instance.dict_trained_models[classifier + " + " + vectorizer][
        "trained_model"
    ]

    return model.predict(com_vectorized)
