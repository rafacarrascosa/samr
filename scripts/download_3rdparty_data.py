import os
import urllib.request

import nltk

from samr.settings import DATA_PATH

# Create data folder if necessary
if not os.path.isdir(DATA_PATH):
    print("Creating data folder at {}".format(DATA_PATH))
    os.makedirs(DATA_PATH)
else:
    print("Data folder found at {}".format(DATA_PATH))

# Download inquirer data
filename = os.path.join(DATA_PATH, "inquirerbasicttabsclean")
url = "http://www.wjh.harvard.edu/~inquirer/inqtabs.txt"
if not os.path.isfile(filename) or os.stat(filename).st_size != 2906024:
    print("Downloading {} into {}".format(url, filename))
    urllib.request.urlretrieve(url, filename)
else:
    print("Harvard Inquirer lexical data found at {}".format(filename))

# Download nltk data
nltk.download("wordnet")
nltk.download("punkt")

print("\n3rd party data downloaded correctly.\n")
