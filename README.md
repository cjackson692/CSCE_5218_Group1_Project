# CSCE_5218_Group1_Project - English to Tagalog - NMT

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


## Repository description
```
 ├── Datasets # bin for different data
    ├── Clean Data.zip #compressed archive for the data used in this study, after the english removal has been conducted on tagalog sentences (compressed for efficiency)
    ├── LICENSE # Usage license for OpenSubtitles corpus
    ├── README # corpus access information
    ├── data_acquisition.py # downloads the appropriate data from source
    └── en-tl.txt.zip # source data before english has been removed from the tagalog sentences (compressed for efficiency)
├── Testing_Experimentation # bin for any other files
    ├── special_character_analysis.py
    ├── prepare.py
    ├── tain.bin
    └── val.bin
├── README.md
├── TranslationModel.py # Contains the architecture for the translation model used 
├── demo.py # interactive translation for demonstration (see above)
├── englishremover.py # preprocessing example designed to eliminate majority english sentences that occur in the tl data. You do not need to run this for the pipeline, its outputs are already stored in Datasets/Clean Data.zip
├── evaluate.py # script to run evaluation metrics (see above)
├── losscurves.png # visualization of train vs eval loss (NOTE: Predictions are basxed on model state at minimum eval loss (2.758135411204124))
├── pred.pickle # output prediction sentences for the entire test set
├── prepare.py # loads and preps data for training (see above)
├── test_translations.txt # 5 example input/ouput/ground truth examples, for qualitative description
├── train.py # all logic to train and save appropriate output files (see above)
└── truths.pickle # aligned ground truths with preds.pickle for evaluation

```

## Notes
- pr1 (bpe) is stale, was an experimetn in alternate encoding schemas
