Sentiment Analysis on Movie Reviews
===================================

This is an entry to [Kaggle](http://www.kaggle.com/)'s
[Sentiment Analysis on Movie Reviews](http://www.kaggle.com/c/sentiment-analysis-on-movie-reviews) (SAMR)
competition.

It's written for Python 3.3 and it's based on [`scikit-learn`](http://scikit-learn.org/)
and [`nltk`](http://www.nltk.org/).


Problem description
-----------------

Quoting from Kaggle's [description page](http://www.kaggle.com/c/sentiment-analysis-on-movie-reviews):

This competition presents a chance to benchmark your sentiment-analysis ideas
on the [Rotten Tomatoes](http://www.rottentomatoes.com/) dataset. You are asked
to label phrases on a scale of five values: negative, somewhat negative,
neutral, somewhat positive, positive.

Some examples:

 - **4** (positive): _"They works spectacularly well... A shiver-inducing, nerve-rattling ride."_
 - **3** (somewhat positive): _"rooted in a sincere performance by the title character undergoing midlife crisis"_
 - **2** (neutral): _"Its everything you would expect -- but nothing more."_
 - **1** (somewhat negative): _"But it does not leave you with much."_
 - **0** (negative): _"The movies progression into rambling incoherence gives new meaning to the phrase fatal script error."_

So the goal of the competition is to produce an algorithm to classify phrases
into these categories. And that's what `samr` does.


How to use it
-------------

After installing just run:

    generate_kaggle_submission.py samr/data/model2.json > submission.csv

And that will generate a Kaggle submission file that scores near `0.65844` on the
[leaderboard](http://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/leaderboard)
(should take 3 minutes, and as of 2014-07-22 that score is the 2nd place).

The `model2.json` argument above is a configuration file for `samr` that
determines how the `scikit-learn` pipeline is going to be built and other
hyperparameters, here is how it looks:

    {
     "classifier":"randomforest",
     "classifier_args":{"n_estimators": 100, "min_samples_leaf":10, "n_jobs":-1},
     "lowercase":"true",
     "map_to_synsets":"true",
     "map_to_lex":"true",
     "duplicates":"true"
    }

You can try `samr` with different configuration files you make (as long as the
options are implemented), yielding
different scores and perhaps even better scores.

### Just tell me how it works

In particular `model2.json` feeds a [random forest classifier](http://en.wikipedia.org/wiki/Random_forest)
with a concatenation of 3 kinds of features:

 - The [decision functions](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier.decision_function)
   of set of vanilla SGDClassifiers trained in a one-versus-others scheme using
   [bag-of-words](http://en.wikipedia.org/wiki/Bag-of-words_model) as features.
   It's classifier inside a classifier, [yo dawg!](http://i.imgur.com/aueqLyL.png)
 - The decision functions of set of vanilla SGDClassifiers trained in a one-versus-others scheme using bag-of-words
   on the [wordnet](http://wordnetweb.princeton.edu/perl/webwn?s=bank) synsets of the words in a phrase.
 - The amount of "positive" and "negative" words in a phrase as dictated by
   the [Harvard Inquirer sentiment lexicon](http://www.wjh.harvard.edu/~inquirer/spreadsheet_guide.htm)

During prediction, it also checks for duplicates between the training set and
the train set (there are quite a few).

And that's it! Want more details? see the code! it's only 350 lines.


Installation
------------

If you know the drill, this should be enough:

    git clone https://github.com/rafacarrascosa/samr.git
    pip install -e samr -r samr/docs/setup/requirements-dev.txt
    download_3rdparty_data.py

Then you will need to **manually download** `train.tsv` and `test.tsv` from the
competition's [data folder](http://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data)
and unzip them into the `samr/data` folder. You may be asked to join Kaggle and/or
accept the competition rules before downloading the data.

Even though `samr` is writen for Python 3.3 it may also work with Python 2.7
(and the last time I checked it was), but this is not supported and it may
break in the future.

If the short instructions are not enough, read on.


### Full instructions for Ubuntu

These instructions will install the development version of `samr` inside a
Python 3.3 virtualenv and were thought for a blank, vanilla Ubuntu 14.04 and
tested using [Docker](https://www.docker.com/) (awesome tool btw). They should
work more or less unchanged with other Ubuntu versions and Debian-based OSs.

Open a console and 'cd' into an empty folder of your choice. Now, execute the
following commands:

Install python 3.3 and compilation requirements for numpy and scipy:

    sudo apt-get update
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:fkrull/deadsnakes
    sudo apt-get update
    sudo apt-get install -y python3.3 python3.3-dev python-scipy gfortran libopenblas-dev liblapack-dev git wget

Create virtualenv, bootstrap pip and boostrap numpy:

    python3.3 -m venv venv
    source venv/bin/activate
    wget https://bootstrap.pypa.io/get-pip.py
    python3.3 get-pip.py
    echo 'PATH="$VIRTUAL_ENV/local/bin:$PATH"; export PATH' >> venv/bin/activate
    source venv/bin/activate
    pip install numpy==1.8.1

Clone and install samr:

    git clone https://github.com/rafacarrascosa/samr.git
    pip install -e samr -r samr/docs/setup/requirements-dev.txt
    download_3rdparty_data.py

Optionally run the tests:

    nosetests samr/tests

Lastly, you will need to   **manually download** `train.tsv` and `test.tsv` from the
competition's [data folder](http://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data)
and unzip them into the `samr/data` folder. You may be asked to join Kaggle and/or
accept the competition rules before downloading the data.

The installation is self-contained (within the folder you chose at the start) with
two exceptions:

- Lines starting with `sudo apt-get` made system-wide changes, to uninstall
  those you will to use `sudo apt-get remove`.
- `nltk` downloads data to `~/nltk_data`, once you don't use `nltk` it's safe
  to erase that folder.


Licensing
---------

This project is open-source and BSD licensed, see the LICENSE file for details.

This license basically allows you to do anything, but in case you're wondering:
I'm ok if you use `samr` to beat my score at the competition, just share back
what you've learned!


Credits
---------

This project was developed by Rafael Carrascosa, you can contact me at
<rafacarrascosa@gmail.com>.

