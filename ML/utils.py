import os
import re

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