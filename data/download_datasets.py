# !wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip"
# !wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.names"
# !wget https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip

# import copy
import os
from subprocess import call

print("")

print("Downloading UCI HAR Dataset...")
if not os.path.exists("UCI HAR Dataset.zip"):
    call(
        'wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip"',
        shell=True
    )
    print("Downloading done.\n")
else:
    print("Dataset already downloaded. Did not download twice.\n")


print("Extracting...")
extract_directory = os.path.abspath("UCI HAR Dataset")
if not os.path.exists(extract_directory):
    call(
        'unzip -nq "UCI HAR Dataset.zip"',
        shell=True
    )
    print("Extracting successfully done to {}.".format(extract_directory))
else:
    print("Dataset already extracted. Did not extract twice.\n")


print("Downloading opportunity dataset...")
if not os.path.exists("OpportunityUCIDataset.zip"):
    call(
        'wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip"',
        shell=True
    )
    print("Downloading done.\n")
else:
    print("Dataset already downloaded. Did not download twice.\n")


print("Extracting...")
if not os.path.exists("oppChallenge_gestures.data"):
    from preprocess_data import generate_data
    generate_data("OpportunityUCIDataset.zip", "oppChallenge_gestures.data", "gestures")
    print("Extracting successfully done to oppChallenge_gestures.data.")
else:
    print("Dataset already extracted. Did not extract twice.\n")
