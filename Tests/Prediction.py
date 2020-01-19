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

# Load model
from sources.utils import load_object_pickle

file_name_model = "model.pkl"
path_model = os.path.join(os.environ["Data_Processed"], file_name_model)
model = load_object_pickle(path_model)

# %%
from sources.prediction import test_model
import numpy as np

raw_comment = np.array(
    [
        "Mot de passe",
        "Digital",
        "offre fibre+ envoi code de connexion.",
        "le mot de passe provisoire pour ce compte orange a été envoyé avec succès par courrier postal au 17 pl 9 octobre 59400 cambrai france.",
        "telephone",
        "le client a perdu ses ids de connexion",
    ]
)
test_model(model, "LinearSVC", "count vector", raw_comment)

# %%
file_name_sample = "sample_base_apprentissage.csv"
df_sample = pd.read_csv(os.path.join(os.environ["Data_Sample_Raw"], file_name_sample))

# %%
df_sample.head()

# %%
df_sample[df_sample["Motif"] == 1]

# %%
