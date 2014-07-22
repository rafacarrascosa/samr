import os
import urllib.request

import nltk

from samr.settings import DATA_PATH


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
