# BankruptcyBenchmarkBaselines

**NOTE: A new, easy to use version of this repository with access to the data will be released soon (expected November 2023)**

This repository contains the code that can be used to reconstruct the benchmark &amp; baselines for bankruptcy prediction using textual data. The corresponding paper was presented at the FinNLP workshop @ IJCAI-ECAI 2022 and published in the ACL Anthology.

Link to the paper: https://arxiv.org/abs/2208.11334

**Citation**: Arno, H., Mulier, K., Baeck, J., Demeester, T.: Next-Year Bankruptcy Prediction from Textual Data: Benchmark and Baselines. Proceedings of the Fourth Workshop on Financial Technology and Natural Language Processing FinNLP@IJCAI2022, pp. 36–42, (2022)

## USAGE

### Prerequisites
- The LoPucki BRD case table (https://lopucki.law.ufl.edu/index.php): Contains information on bankruptcy filings under chapter 7 or chapter 11 of the US bankruptcy code for public companies in the US. The dataset is no longer updated  as of 2023 but is free of charge.
- EDGAR-CORPUS (https://arxiv.org/abs/2109.14394): A corpus of 10-k reports from all public companies in the US for over 25 years. The authors also provide a crawling tool to keep the corpus up-to-date. This can be accessed through the paper. In this repo we focus on the MD&A section (Management Discussion & Analysis).

An environment.yml file is added to the repository which can be used to install the required dependencies. Note that the file does contain more dependencies than strictly needed to run the code. 

### Workflow

In each folder there are script that can be used to perform the steps described in the paper and specify the parameters (such as the location of the files, hyperparameters, time periods to use, ...). Run the scripts in each folder in the following order:


- **Data_Source_Integration**: Links both data sources and stores the filings of healthy companies, companies that ultimatly went bankrupt and the dates on which they filed for bankruptcy. (THIS STEP MIGHT TAKE A WHILE)
- **Prep-Bag-of-Words**: Prepare the data for training the Bag-of-Words models. Execute in the order indicated by the number in the name of the files. NOTE: Some steps are memory intensive, if the execution of a script fails when using large amounts of data (e.g. when using 2000 - 2020 data without sampling), it is suggested to execute the script in blocks indicated by the dashes. This folder also contains the code to train the word2vec model that will be used in a later stage. For more details on which steps are performed, consult the dedicated section in the paper.
- **PrepTransformer**: Prepare the data for training the transformer model. Note that the amount of preprocessing is limited compared to the Bag-of-Words models.
- **Sampling**: Link each processed document to an id. Next, perform sampling as described in the paper based on this document id. The resulting datasets contain time-agnostic firm-year samples than can be used to build the models (train_dev, train_full, dev and holdout - taking 1 or 3 documents into account for prediction).
- **Encoding**: The documents, processed in the previous steps and associated with a document id, are encoded in two possible ways. Either the trained word2vec model is used to compute the mean embedding of a document, or the document is passed through the LongFormer model of Beltagy et al (2020) with the help of the Huggingface API. Afterwards, we store the document representations, which are used in a later stage to perform the classification. The representations are static and finetuning the transformer-based model will be part of future work and is not available right now.
- **Modelling/Binary_TFIDF**: Training, evaluation and hyperparameter optimisation of the models using binary or term frequency-inverse document frequency (TF-IDF) features with a logistic regression classifier. The sklearn library was used and the hyperparameters ranges can be specified in the script.
- **Modelling/W2V_Transformers**: Training and evaluation of the models using word2vec or Longformer encodings as document representation and a feed-forward neural network classifier. The PyTorch library was used, while hyperparameter optimisation was performed with Optuna (the code for hyperparameter optimisation is not added and will be made avaialbe upon request).

For questions, comments or recommendations: henri.arno@ugent.be
