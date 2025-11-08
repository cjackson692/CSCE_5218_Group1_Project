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
  - doesn't do anything yet

run demo.py
  - loads model
  - creates a interactive CLI where you can type out a sequence, and get a tagalog translation
