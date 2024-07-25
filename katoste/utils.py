import logging
import os
import pickle
import shutil
import numpy as np
import re

from pathlib import Path

class FormatError(Exception):
    """Exception raised for errors in the input format."""
    def __init__(self, message):
        super().__init__(message)

def check_cell_string(cell='r1[2:27]'):
    """
    Validates and parses the 'cell' string parameter to ensure it
    follows the expected format and extracts the read group and index range.

    Parameters:
        cell (str): A string specifying the read group and index range 
        in the format 'r1[start:end]' or 'r2[start:end]'. Default is 'r1[2:27]'.

    Returns:
        tuple: A tuple containing the read group (str) and the start 
        (int) and end (int) indices parsed from the 'cell' string.

    Raises:
        FormatError: If the 'cell' string does not match the expected format.
    """
    match = re.match(r'(r[12])\[(\d+):(\d+)\]', cell)
    if not match:
        raise FormatError("Cell format must be 'r1[start:end]' or 'r2[start:end]'")

    read_group, start, end = match.groups()
    start, end = int(start), int(end)

    return read_group, start, end

def safety_check_eval(s, danger="();."):
    chars = set(list(s))
    if chars & set(list(danger)):
        return False
    else:
        return True
    
def get_module_path():
    import pathlib, katoste
    return pathlib.Path(katoste.__file__).resolve().parent

def save_pickle(obj, file_path):
    """
    Save an object to a pickle file.

    Parameters:
        obj (any): The object to be saved.
        file_path (str): The path to the pickle file.

    Returns:
        None
    """
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(file_path):
    """
    Load an object from a pickle file.

    Parameters:
        file_path (str): The path to the pickle file.

    Returns:
        any: The loaded object.
    """
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def check_file_exists(f, except_when=None) -> bool:
    """
    Check whether the file exists.

    Args:
        f (str): Path to the input file.
        except_when (bool): Throw exception when file exists (or not). Default: None

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    _path_exists = os.path.exists(f)
    if except_when is not None and except_when == _path_exists:
        raise FileNotFoundError(f"The file '{f}' does {'not ' if not except_when else ''}exist")

    return _path_exists


def check_directory_exists(path, except_when=None) -> bool:
    """
    Check if a file exists, or if its parent directory exists.

    Parameters:
        path (str): Path to the file or directory.
        except_when (bool): Throw exception when file exists (or not). Default: None

    Returns:
        bool: True if the parent directory exists or if the file exists, False otherwise.
    """
    _ret_val = False
    if not os.path.isfile(path):
        _ret_val = os.path.exists(path)
    else:
        path = os.path.dirname(path)
        # handle file created in the same directory
        if path == "":
            _ret_val = True
        else:
            _ret_val = os.path.exists(path)
    
    if except_when is not None and except_when == _ret_val:
        raise FileNotFoundError(f"The directory '{path}' does {'not ' if not except_when else ''}exist")
    
    return _ret_val


def check_adata_structure(f):
    """
    Check the validity of the input Open-ST h5 object.

    Args:
        f (str): Path to the input Open-ST h5 object.

    Raises:
        KeyError: If required properties are not found in the file.
    """
    import h5py

    with h5py.File(f, "r") as file:
        if "obsm/spatial" not in file:
            raise KeyError("The Open-ST h5 object does not have the 'obsm/spatial' property.")

        if "obs/tile_id" not in file:
            raise KeyError("The Open-ST h5 object does not have the 'obs/tile_id' property.")

        if "obs/total_counts" not in file:
            raise KeyError("The Open-ST h5 object does not have the 'obs/total_counts' property.")

        if "spatial_aligned" in file:
            logging.warn("The Open-ST h5 object has a 'spatial_aligned' layer")


def load_properties_from_adata(f, properties: list = ["obsm/spatial"], backed: bool=False) -> dict:
    """
    Load specified properties from an AnnData file (h5py format).

    Args:
        f (str): Path to the AnnData h5py file.
        properties (list, optional): List of property paths to load from the file.
        backed (bool, optional): If True, data will not be read into memory.

    Returns:
        dict: A dictionary containing the loaded properties.
            - For each property path specified in the 'properties' list:
                * The dictionary key is the property path.
                * The value is the corresponding parsed property data.

    Notes:
        - This function loads specified properties from an AnnData h5py file.
        - The 'properties' list should consist of property paths within the file.
        - Returns a dictionary where keys are property paths and values are the loaded data.
    """
    import h5py
    from anndata import AnnData
    from anndata._io.specs import read_elem

    parsed_properties = {}

    if isinstance(f, AnnData):
        for p in properties:
            parsed_properties[p] = read_elem(f[p])
    elif isinstance(f, str):
            if backed:
                _f = h5py.File(f)
                for p in properties:
                    parsed_properties[p] = _f[p]
            else:
                with h5py.File(f) as _f:
                    for p in properties:
                        parsed_properties[p] = read_elem(_f[p])
    else:
        raise TypeError("Type of 'f' is incorrect. It needs to be an AnnData or str object.")

    return parsed_properties


def check_obs_unique(adata, obs_key: str = "tile_id") -> bool:
    """
    Check if the values in a specified observation key in an AnnData object are unique.

    Args:
        adata (AnnData): AnnData object to check for unique observations.
        obs_key (str, optional): The name of the observation key to check for uniqueness. Defaults to "tile_id".

    Returns:
        bool: True if the specified observation key has unique values, False otherwise.

    Raises:
        ValueError: If the specified observation key exists in the AnnData object but is not unique.
    """
    return adata.obs[obs_key].nunique() == 1


def copytree2(source: str, dest: str) -> str:
    """
    Recursively copy the contents of a source directory to a destination directory.

    Args:
        source (str): The source directory to be copied.
        dest (str): The destination directory where the contents will be copied to.

    Returns:
        str: The path to the destination directory where the contents were copied.

    Notes:
        - This function creates the destination directory and its parent directories if they do not exist.
        - It checks if the source and destination directories already exist and have the same size.
          If so, it skips copying.
        - If the source and destination directories differ in size or do not exist, it performs a recursive copy.

    """
    Path(dest).mkdir(parents=True, exist_ok=True)
    dest_dir = os.path.join(dest, os.path.basename(source))
    if os.path.exists(dest_dir) and os.path.getsize(dest_dir) == os.path.getsize(source):
        print("The directory {OUTFILE} was already copied. Skipping!")
    else:
        shutil.copytree(source, dest_dir, dirs_exist_ok=True)
    return dest_dir

def get_package_path() -> str:
    """Get the absolute path of the directory containing the current Python package.

    Returns:
        str: Absolute path of the directory containing the current Python package.
    """
    import openst
    return os.path.dirname(os.path.abspath(openst.__file__))

def get_absolute_package_path(relative_path) -> str:
    """
    Get the absolute path by concatenating the package path and the relative path.

    Args:
        relative_path (str): Relative path from the package directory.

    Returns:
        str: Absolute path.
    """
    package_path = get_package_path()
    return os.path.join(package_path, relative_path)

def h5_to_dict(adata) -> dict:
    """
    Recursively converts an h5py.Group object and its nested datasets into a nested dictionary structure.

    Parameters:
        adata (h5py.Group): An h5py Group object to be converted.

    Returns:
        dict: A nested dictionary representing the structure of the h5py Group object.
            Leaf nodes contain strings representing the type and shape (if applicable) of the datasets.
            Non-leaf nodes contain nested dictionaries representing their child groups and datasets.

    Notes:
        - Leaf nodes in the resulting dictionary contain strings formatted as "{type}_{shape}".
          If the dataset has no shape attribute (e.g., scalar dataset), shape will be None.
          Example: "<class 'numpy.ndarray'>_(10,)"
        - Non-leaf nodes in the resulting dictionary contain nested dictionaries
          representing their child groups and datasets.
    """
    import h5py

    result = {}
    for key, value in adata.items():
        if isinstance(value, h5py.Group):
            result[key] = h5_to_dict(value)
        else:
            dataset_type = str(type(value))
            dataset_shape = value.shape if hasattr(value, 'shape') else None
            result[key] = f"{dataset_type}_{dataset_shape}"
    return result

def write_key_to_h5(adata, key, data, delete_before=False):
    if key in adata and not delete_before:
        adata[key][:] = data
    elif key in adata and delete_before:
        del adata[key]
    else:
        adata[key] = data

def binary_search(arr, low, high, x):
    if high >= low:
        mid = (high + low) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] > x:
            return binary_search(arr, low, mid - 1, x)
        else:
            return binary_search(arr, mid + 1, high, x)
 
    else:
        return -1
    
def group_intervals(arr, min_interval):
    arr = np.sort(arr)
    
    intervals = []
    start = arr[0]
    end = arr[0]
    
    for i in range(1, len(arr)):
        if arr[i] - end > min_interval:
            intervals.append((start, end))
            start = arr[i]
        
        end = arr[i]
    
    intervals.append((start, end))
    
    return intervals