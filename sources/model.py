import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.base import clone
import matplotlib.pylab as plt
from IPython.display import display
import itertools
import numpy as np


class Model:
    def __init__(
        self,
        df: pd.core.frame.DataFrame,
        raw_x_column_name,
        x_column_name,
        y_column_name,
    ):
        self.__corpus = df[[raw_x_column_name, x_column_name, y_column_name]].rename(
            columns={
                raw_x_column_name: "raw_comment",
                x_column_name: "comment",
                y_column_name: "pattern",
            }
        )

        self.__x = self.__corpus["comment"]
        self.__y = self.__corpus["pattern"]

        train, test = train_test_split(self.__corpus, test_size=0.1)

        self.x_train = train["comment"].fillna("erreur")
        self.x_test = test["comment"].fillna("erreur")

        self.y_train = train["pattern"].fillna("erreur")
        self.y_test = test["pattern"].fillna("erreur")

        self.x_train_vect = dict()
        self.x_test_vect = dict()

        self.vect_types = []
        self.vectorizer = dict()
        self.__vectorize_count()
        self.__vectorize_tf_idf()

        self.dict_trained_models = dict()

        self.summary = pd.DataFrame(
            columns=["model", "training set accuracy", "test set accuracy"]
        )

    def fill_dict(self, classifier_name, classifier):
        for vect_type in self.vect_types:
            self.__fill_dict(classifier_name, clone(classifier), vect_type)

    def display_misclassification(self, classifier):
        df = self.dict_trained_models[classifier]["errors"]
        pd.set_option("display.max_rows", df.shape[0] + 1)
        print(df)

    def plot_confusion_matrix(self, classifier, normalize=False):
        dict_characters = {0: "Non-digital", 1: "digital"}
        classes = list(dict_characters.values())
        cm = self.dict_trained_models[classifier]["confusion_matrix"]
        Model.__plot_confusion_matrix(cm, classes, normalize, title_addition=classifier)

    @classmethod
    def __plot_confusion_matrix(cls, cm, classes, normalize=False, title_addition=""):
        plt.figure(figsize=(10, 5))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.get_cmap("pink"))
        plt.title("Confusion matrix " + title_addition)
        plt.colorbar()
        tick_marks = np.array([0.0, 1.0])
        plt.xticks(tick_marks, classes, rotation=0)
        plt.yticks(tick_marks, classes)
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        thresh = cm.max() / 2.0
        for i, j in itertools.product([0, 1], [0.25, 0.75]):
            plt.text(
                i,
                j,
                cm[i, round(j)],
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, round(j)] < thresh else "black",
            )
        plt.tight_layout()
        plt.ylabel("Actual")
        plt.xlabel("Predicted")

    def __vectorize_count(self):
        vect = CountVectorizer(min_df=5, max_df=0.8)
        vect.fit(self.x_train)
        self.x_train_vect["count vector"] = vect.transform(self.x_train)
        self.x_test_vect["count vector"] = vect.transform(self.x_test)
        self.vect_types.append("count vector")
        self.vectorizer["count vector"] = vect

    def __vectorize_tf_idf(self):
        vect = TfidfVectorizer(min_df=3, max_df=0.9)
        vect.fit(self.x_train)
        self.x_train_vect["tf-idf vector"] = vect.transform(self.x_train)
        self.x_test_vect["tf-idf vector"] = vect.transform(self.x_test)
        self.vect_types.append("tf-idf vector")
        self.vectorizer["tf-idf vector"] = vect

    def __fill_dict(self, classifier_name, classifier, vect_type):

        y_train = self.y_train
        y_test = self.y_test

        x_train_vect = self.x_train_vect[vect_type]
        x_test_vect = self.x_test_vect[vect_type]

        classifier.fit(x_train_vect, y_train)

        key = str(classifier_name) + " + " + str(vect_type)
        y_pred = classifier.predict(x_test_vect)
        test_accuracy = classifier.score(x_test_vect, y_test)
        train_accuracy = classifier.score(x_train_vect, y_train)

        self.dict_trained_models[key] = dict()
        self.dict_trained_models[key]["accuracy"] = {
            "test": test_accuracy,
            "train": train_accuracy,
        }
        self.dict_trained_models[key]["trained_model"] = classifier
        self.dict_trained_models[key]["errors"] = self.__misclassification_table(y_pred)
        self.dict_trained_models[key]["confusion_matrix"] = confusion_matrix(
            y_test, y_pred
        )

        new_result_data = {
            "model": [key],
            "training set accuracy": [train_accuracy],
            "test set accuracy": [test_accuracy],
        }
        df_new_result = pd.DataFrame(new_result_data)
        self.summary = self.summary.append(df_new_result, ignore_index=True)

    def __misclassification_table(self, y_pred):
        Com = []
        Com_treated = []
        pred = []
        real = []
        for inputs, input_treated, prediction, label in zip(
            self.__corpus["raw_comment"][self.x_test.index],
            self.__corpus["comment"][self.x_test.index],
            y_pred,
            self.y_test,
        ):
            if prediction != label:
                Com.append(inputs)
                Com_treated.append(input_treated)
                pred.append(prediction)
                real.append(label)

        data = {"Com": Com, "Com_treated": Com_treated, "pred": pred, "real": real}
        error = pd.DataFrame(data)
        return error
