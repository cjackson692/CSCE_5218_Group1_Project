# CSCE_5218_Group1_Project

## Full Pipeline

run prepare.py
  - takes data from 'Cleaned Data.zip'
  - tokenize, encode, crop, and save datasets and tokenizers to file

run train.py
  - takes output of prepare.py, preps the data the rest of the way for training, trains model, and saves the following
    - model state_dict
    - aligned prediction and truth files (as pickle files, they are lists of lists
    - train and test dataloader
    - losscurves plot
  - prints five random example translations
  - our training time: 9hrs 30 min

run evaluate.py
  - dutilizes truths.pickle and preds.pickle yo perform evaluation

run demo.py
  - loads model
  - if you don't want to train the model yourslef, download the pretrained state dict from here, the demo can load from the file: https://drive.google.com/file/d/1lyb5HHtMsnMCINKlYCfcKlYqx8dHLF8-/view?usp=sharing (fille too large for github)
  - creates a interactive CLI where you can type out a sequence, and get a tagalog translation
