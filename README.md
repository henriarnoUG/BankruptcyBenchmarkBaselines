# BankruptcyBenchmarkBaselines
This repository contains the code that can be used to reconstruct the benchmark &amp; baselines for bankruptcy prediction using textual data. The corresponding paper was presented at the FinNLP workshop @ IJCAI-ECAI 2022 and published in the ACL Anthology.

The code will be released stepwise (estimated to be complete by end of August).

(Temporary) link to the paper: https://mx.nthu.edu.tw/~chungchichen/FinNLP2022_IJCAI/6.pdf

## USAGE

### Prerequisites
- The LoPucki BRD case table (https://lopucki.law.ufl.edu/index.php)
- EDGAR-CORPUS (https://arxiv.org/abs/2109.14394)


### LOGIC

In each folder there is a main script that can be used to specify the parameters (such as the location of the files, where to store the output, periods, ...) and perform a step described in the paper.Run the scripts in each folder in the following order:


- **Data_Source_Integration**: Link both data sources and stores the filings of healthy companies, companies that ultimatly went bankrupt and the dates on which they filed for bankruptcy. (THIS STEP MIGHT TAKE A WHILE)
- **BoW / Transformer (2 folders)**: Prepare the data for training the Bag-of-Words / Transformer models. The steps in these folders are split up in 1) preprocessing and 2) sampling. Execute in the order indicated by the name of the files. NOTE: Some steps are memory intensive, if the execution of a script fails (e.g. 1_initial_processing.py), it is suggested to execute the script in blocks indicated by the dashes.
