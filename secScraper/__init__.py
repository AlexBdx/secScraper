from os.path import dirname, basename, isfile, join, abspath, split
import glob

path_glob = join(dirname(abspath(__file__)), '*.py')
modules = glob.glob(path_glob)
module_blacklist = [
    '__init__.py',
    'setup.py',
    'Filing metrics.py',
    'main.py',
    'ec2_scraping.py'
]

__name__ = "secScraper"
__version__ = "0.0.30"
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and split(f)[1] not in module_blacklist]
# print("[INFO] Loaded the following modules:\n", __all__)
