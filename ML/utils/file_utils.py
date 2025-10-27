import os
import re
from pathlib import Path


def find_latest_version_path(base_path):
    directory, full_filename = os.path.split(base_path)
    if not directory:
        directory = '.'
    filename, extension = os.path.splitext(full_filename)
    latest_version = -1
    latest_file_path = None
    regex = re.compile(f"{re.escape(filename)}_v(\\d+){re.escape(extension)}")
    
    try:
        for f in os.listdir(directory):
            match = regex.match(f)
            if match:
                version = int(match.group(1))
                if version > latest_version:
                    latest_version = version
                    latest_file_path = os.path.join(directory, f)
    except FileNotFoundError:
        return None
    
    if not latest_file_path and os.path.exists(base_path):
        return base_path
    return latest_file_path


def get_next_version_path(base_path):
    directory, full_filename = os.path.split(base_path)
    filename, extension = os.path.splitext(full_filename)
    version = 1
    while True:
        versioned_filename = f"{filename}_v{version}{extension}"
        next_path = os.path.join(directory, versioned_filename)
        if not os.path.exists(next_path):
            return next_path
        version += 1


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_model_versions_dir(base_dir="models"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "..", base_dir, "vimaan_nlu_model_best")
    return ensure_directory(models_dir)


def get_latest_model_path():
    models_dir = get_model_versions_dir()
    versions = []
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            if item.startswith('v') and item[1:].isdigit():
                versions.append(int(item[1:]))
    
    if not versions:
        return None
    
    latest_version = max(versions)
    return os.path.join(models_dir, f"v{latest_version}")
