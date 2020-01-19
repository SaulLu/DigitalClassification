# %%
# Config
%load_ext autoreload
%autoreload 2

import os

# Change the content of the text in the path variable to store the path to the main folder
path = "/mnt/storage/Documents/CentralePa/3A/Big Data and AI/DigitalClassification/"

os.chdir(path)
os.environ["Root_DIR"] = path

os.environ["Data_DIR"] = os.path.join(path, "Data")
os.environ["Data_Raw"] = os.path.join(os.environ["Data_DIR"], "Raw")
os.environ["Data_Processed"] = os.path.join(os.environ["Data_DIR"], "Processed")
os.environ["Data_Sample_Raw"] = os.path.join(os.environ["Data_DIR"], "Sample", "Raw")
os.environ["Data_Sample_Processed"] = os.path.join(
    os.environ["Data_DIR"], "Sample", "Processed"
)
os.environ["Data_Ref"] = os.path.join(os.environ["Data_DIR"], "Ref")

# %%
# Import File
import pandas as pd

pd.set_option("display.max_colwidth", -1)

# file_name_sample_pickle = "sample_base_apprentissage_V4.pkl"
# newPath = os.path.join(os.environ["Data_Sample_Processed"],
#                                    file_name_sample_pickle)
# df=pd.read_pickle(newPath)

file_name_full = "learning_base_preprocessed.pkl"
newPath = os.path.join(os.environ["Data_Processed"], file_name_full)
df = pd.read_pickle(newPath)

# Load model
from sources.utils import load_object_pickle

file_name_model = "model.pkl"
path_model = os.path.join(os.environ["Data_Processed"], file_name_model)
model = load_object_pickle(path_model)

# %%
# Modelling
# Init
from sources.model import Model

model = Model(df, "commentaire_brut", "comment_preprossed", "Motif")

# %%
# add new models
from sklearn.linear_model import SGDClassifier

SGD = SGDClassifier(max_iter=5, tol=None)
model.fill_dict("Stochastic Gradient Descent", SGD)

from sklearn.svm import LinearSVC

LSVC = LinearSVC()
model.fill_dict("LinearSVC", LSVC)

from xgboost import XGBClassifier

XGB = XGBClassifier()
model.fill_dict("Xgboost", XGB)

from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier()
model.fill_dict("Random Forest Classifier", RFC)

# %%
# save model in pickle file
from sources.utils import save_object_pickle

file_name_full = "model.pkl"
newPath = os.path.join(os.environ["Data_Processed"], file_name_full)
save_object_pickle(newPath, model)

# %%
from sources.model import Model

print(f"test set size : {model.x_test.size}, train set size : {model.x_train.size}")
# %%
# Summary
model.summary
#%%
# show results
for classifier in list(model.dict_trained_models.keys()):
    # print('------------')
    # print(f'Classifier : {classifier}')
    model.plot_confusion_matrix(classifier)
    # model.display_misclassification(classifier)
    # print(model.dict_trained_models[classifier]['accuracy'])


# %%
