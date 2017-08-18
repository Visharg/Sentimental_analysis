# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 11:44:33 2017

@author: Visharg Shah
"""

# Miscellaneous imports
from os import getcwd, listdir, remove
from os.path import join, isfile, isdir
import wget
import tarfile


class Condense_Reviews(object):
    def __init__(self, data_dir, output_dir):
        self.output_dir = output_dir
        self.data_dir = data_dir

    def get_data(self):
        # Download data
        url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        fname = wget.download(url)

        # Unpack data
        tar = tarfile.open(fname, "r:gz")
        tar.extractall(path=self.output_dir)
        tar.close()

        # Remove leftover gz file
        remove(fname)


    def condense(self):
        # Organize subdirectories by sentiment
        pos_train = [join(self.data_dir, "train", "pos")]
        pos_test =  [join(self.data_dir, "test", "pos")]
        neg_train = [join(self.data_dir, "train", "neg")]
        neg_test =  [join(self.data_dir, "test", "neg")]
        unlab_train = [join(self.data_dir, "train", "unsup")]

        # Define dictionary of sentiments paired with associated subdirectories
        sent_dict = {"pos-train": pos_train,
                     "pos-test": pos_test,
                     "neg-train": neg_train,
                     "neg-test": neg_test,
                     "unlab-train": unlab_train}

        # Condense all files from positive sentiment subdirectories
        for sentiment, subdirs in sent_dict.items():
            # Define full path for output file
            full_fname = join(self.output_dir, sentiment + ".txt")
            with open(full_fname, "w",encoding="utf8") as output_file:
                for _dir in subdirs:
                    for _file in listdir(_dir):
                        for line in open(join(_dir, _file),encoding="utf8"):
                            output_file.write(line)
                            output_file.write("\n")


def run_condenser():
    # If data is not yet condensed, condense it
    if (not isfile("pos.txt")) and (not isfile("neg.txt")) and (not isfile("unlab.txt")):
        # Initialize a Condense_Reviews object
        output_dir = getcwd()
        data_dir = join(getcwd(), "aclImdb")

        # Condense reviews
        condenser = Condense_Reviews(data_dir, output_dir)

        if isdir("aclImdb"):
            print("Gathering data...")
            condenser.condense()
        else:
            print("The data directory seems to be missing. Downloading data now...")
            condenser.get_data()

            print("\nGathering data...")
            condenser.condense()

    else:
        print("Data is already condensed")


if __name__ == "__main__":
    run_condenser()