import pandas as pd
import ujson
import pickle


def csv_creation(path_excel, path_csv, Includeindex=False):
    """
    Save the excel file at path_excel as csv file at the path_csv
    """
    df = pd.read_excel(path_excel)
    df.to_csv(path_csv, index=Includeindex)


def csv_sample_creation(path, newPath, sample=0.4):
    """
    Create a sample of the original dataset
    """
    df = pd.read_csv(path)
    # Sample extraction of the dataset to help us build our functions
    df = df.sample(frac=sample, replace=True, random_state=1)
    df.to_csv(newPath, index=False)


def json_file_dict(file_path: str):
    """
    """
    with open(file_path, "r") as f:
        json_text = f.read()
    return ujson.loads(json_text)


def save_object_pickle(filename, instance):
    with open(filename, "wb") as output:
        pickle.dump(instance, output, pickle.HIGHEST_PROTOCOL)


def load_object_pickle(filename):
    infile = open(filename, "rb")
    instance = pickle.load(infile)
    infile.close()
    return instance
