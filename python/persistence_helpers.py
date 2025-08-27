import pandas as pd

def save_dataframe_dict_to_csv(df_dict, file_path):
    """
    Save a dictionary of pandas DataFrames into a single CSV file.
    Adds a __key__ column to distinguish which DataFrame each row came from.
    """
    frames = []
    for key, df in df_dict.items():
        df_copy = df.copy()
        df_copy["__key__"] = key  # tag with dictionary key
        frames.append(df_copy)

    big_df = pd.concat(frames, ignore_index=True, sort=False)
    big_df.to_csv(file_path, index=False)


def load_dataframe_dict_from_csv(file_path):
    """
    Load a dictionary of pandas DataFrames from a single CSV file
    saved by save_dataframe_dict_to_csv.
    """
    big_df = pd.read_csv(file_path)
    df_dict = {key: df.drop(columns="__key__") 
               for key, df in big_df.groupby("__key__")}
    return df_dict

import hashlib

def file_hash(path, algo="sha256", chunk_size=8192):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def files_identical(file1, file2):
    return file_hash(file1) == file_hash(file2)

if __name__ == "__main__":

    FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported"

    file1 = "batch_0_depth_map_MOD_depth_anything_v2_large_0.bin"
    file2 = "batch_0_depth_map_MOD_depth_anything_v2_base_0.bin"
    print(files_identical(FILE_PATH + "\\" + file1, FILE_PATH + "\\" + file2))

