# BankruptcyBenchmarkBaselines
This repository contains the code that can be used to reconstruct the benchmark &amp; baselines for bankruptcy prediction using textual data. The corresponding paper was presented at the FinNLP workshop @ IJCAI-ECAI 2022 and published in the ACL Anthology.

(Temporary) link to the paper: https://mx.nthu.edu.tw/~chungchichen/FinNLP2022_IJCAI/6.pdf

**Citation**: Arno, H., Mulier, K., Baeck, J., Demeester, T.: Next-year bankruptcy prediction from textual data: Benchmark and baselines. Proceedings of the Fourth Workshop on Financial Technology and Natural Language Processing FinNLP@IJCAI2022, pp. 36–42, (2022)

## USAGE

### Prerequisites
- The LoPucki BRD case table (https://lopucki.law.ufl.edu/index.php): Contains information on bankruptcy filings under chapter 7 or chapter 11 of the US bankruptcy code for public companies in the US. The dataset has to be purchased and is updated monthly.
- EDGAR-CORPUS (https://arxiv.org/abs/2109.14394): A corpus of 10-k reports from all public companies in the US for over 25 years. The authors also provide a crawling tool to keep the corpus up-to-date. This can be accessed through the paper. In this repo we focus on the MD&A section (Management Discussion & Analysis).

An environment.yml file is added to the repository which can be used to install the required dependencies. Note that the file does contain more dependencies than strictly needed to run the code. 

### Workflow

In each folder there are script that can be used to specify the parameters (such as the location of the files, where to store the output, time periods to use, ...) and perform the steps described in the paper. Run the scripts in each folder in the following order:


- **Data_Source_Integration**: Link both data sources and stores the filings of healthy companies, companies that ultimatly went bankrupt and the dates on which they filed for bankruptcy. (THIS STEP MIGHT TAKE A WHILE)
- **Transformer/PreProcessing**: Prepare the data for training the transformer model. Less preprocessing compared to BoW models.
- **Bag-of-Words/PreProcessing**: Prepare the data for training the Bag-of-Words models. Execute in the order indicated by the number in the name of the files. NOTE: Some steps are memory intensive, if the execution of a script fails when using large amounts of data (e.g. when using 2000 - 2020 data without sampling), it is suggested to execute the script in blocks indicated by the dashes. This folder also contains the code to train the W2V model that will be used in a later stage.
- **Sampling**: Link each processed document to an id. Next, perform sampling as described in the paper based on this document id. The resulting datasets contain time-agnostic firm-year samples than can be used to build the models (train_dev, train_full, dev and holdout - taking 1 or 3 documents into account for prediction).
- **Encoding**: The documents, processed in the previous steps and associated with a doc_id, are encoded in two possible ways. Either the trained word2vec model is used to compute the mean embedding of a document, or the document is passed through the LongFormer model of Beltagy et al with the help of the Huggingface API. Afterwards, we store the document representations, which are used in a later stage to perform the classification. The representations are static and finetuning the transformer-based model will be part of future work and is not available right now.
- **Modelling/Binary_TFIDF**: Training, evaluation and hyperparameter optimisation of the models using binary or TF-IDF features with a logistic regression classifier. The sklearn library was used and the ranges for the hyperparameters can be specified in the script.
- **Modelling/W2V_Transformers**: Training and evaluation of the models using W2V or Longformer encodings as document representation and a FFNN classifier. The PyTorch library was used, while hyperparameter optimisation was performed with Optuna (the code for hyperparameter optimisation is not added and will be made avaialbe upon request - please specify in the GitHub issue section).

For questions, comments or recommendations: henri.arno@ugent.be
