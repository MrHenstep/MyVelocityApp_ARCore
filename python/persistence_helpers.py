import pandas as pd

def save_dataframe_dict_to_csv(df_dict, file_path):
    """
    Saves a dictionary of pandas DataFrames to a single CSV file.
    Each DataFrame in the dictionary is tagged with its corresponding key in a new column named '__key__'.
    All DataFrames are concatenated into one DataFrame and written to the specified CSV file.
    Args:
        df_dict (dict): A dictionary where keys are identifiers and values are pandas DataFrames.
        file_path (str): The path to the CSV file where the combined DataFrame will be saved.
    Returns:
        None
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
    Loads a CSV file into a dictionary of pandas DataFrames, grouped by the '__key__' column.

    Each group in the CSV (as determined by the '__key__' column) becomes a separate DataFrame
    in the returned dictionary. The '__key__' column is removed from each DataFrame.

    Args:
        file_path (str): Path to the CSV file to load.

    Returns:
        dict: A dictionary where keys are the unique values from the '__key__' column,
              and values are pandas DataFrames containing the corresponding group data
              (with the '__key__' column dropped).
    """
    big_df = pd.read_csv(file_path)
    df_dict = {key: df.drop(columns="__key__") 
               for key, df in big_df.groupby("__key__")}
    return df_dict

import hashlib

def file_hash(path, algo="sha256", chunk_size=8192):
    """
    Calculates the hash digest of a file using the specified algorithm.

    Args:
        path (str): Path to the file to be hashed.
        algo (str, optional): Hash algorithm to use (default is 'sha256').
        chunk_size (int, optional): Size of chunks to read from the file in bytes (default is 8192).

    Returns:
        str: Hexadecimal hash digest of the file.

    Raises:
        ValueError: If the specified hash algorithm is not supported.
        FileNotFoundError: If the file at the given path does not exist.
        IOError: If there is an error reading the file.
    """
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def files_identical(file1, file2):
    """
    Compares two files to determine if their contents are identical.

    Args:
        file1 (str): Path to the first file.
        file2 (str): Path to the second file.

    Returns:
        bool: True if the files have identical contents, False otherwise.
    """
    return file_hash(file1) == file_hash(file2)

if __name__ == "__main__":

    FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported"

    file1 = "batch_0_depth_map_MOD_depth_anything_v2_large_0.bin"
    file2 = "batch_0_depth_map_MOD_depth_anything_v2_base_0.bin"
    print(files_identical(FILE_PATH + "\\" + file1, FILE_PATH + "\\" + file2))

