import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-l", "--level", type=int, default=2, help="Version level to increment")
ap.add_argument("-d", "--folder", type=str, default='videotools', help="Package to target")
args = vars(ap.parse_args())
version_level = args["level"]
FOLDER = args["folder"]

if version_level not in [0, 1, 2]:  # Large, medium, small version number
    raise ValueError("The version level can only be 0, 1 or 2")

# I. Update setup.py
# 1. Read setup.py
with open(os.path.join(FOLDER, 'setup.py')) as f:
    # Read the current text
    setup = f.read()
    setup_processing = setup.split('version="')[1]
    text_previous_version = setup_processing.split('"')[0]
    digits_previous_version = [int(v) for v in text_previous_version.split('.')]
    
    # Create a new setup text
    digits_new_version = digits_previous_version
    digits_new_version[version_level] += 1
    text_new_version = [str(v) for v in digits_new_version]
    text_new_version = '.'.join(text_new_version)
    new_version = 'version="' + text_new_version + '"'
    text_setup_py = new_version.join([setup.split('version="')[0], '"'.join(setup_processing.split('"')[1:])])

    #print("[INFO] setup.py:\n", text_setup_py)
# 2. Overwrite setup.py
with open(os.path.join(FOLDER, 'setup.py'), 'w') as f:
    f.write(text_setup_py)


# II. Update __init__.py
# 1. Read __init__.py
with open(os.path.join(FOLDER, '__init__.py')) as f:
    init = f.read()
    init_processing = init.split('__version__ = "')[1]
    init_new_version = '__version__ = "' + text_new_version + '"'
    text_init_py = init_new_version.join([init.split('__version__ = "')[0], '"'.join(init_processing.split('"')[1:])])
    #print("[INFO] __init__.py:\n", text_init_py)

# 2. Overwrite __init__.py
with open(os.path.join(FOLDER, '__init__.py'), 'w') as f:
    f.write(text_init_py)

# III. Bash return
print(text_new_version)  # Bash return
