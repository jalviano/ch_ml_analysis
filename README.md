# Cohere Health ML Analysis

The `lda_analyzer.py` script generates a list of phrases associated with each chief complaint in a set of provided 
medical records. This analysis uses an author-topic model (ATM), an extension of Latent Dirichlet Allocation (LDA) to 
learn topic vectors for "authors" (i.e medical conditions) in the provided corpus.

## Running the model

To build the model and generate the topic vectors, create and source a Python 3 virtual environment from the command 
line in the `ch_ml_analysis` directory:

```
virtualenv -p python3 .env
source .env/activate/bin
```

Next, use `pip` to install the project requirements, which includes a SpaCy language model:

```
pip install -r requirements.txt
```

The first time you build the model, you will need to load the English language stopwords from NLTK. In an IPython shell, 
run:

```
import nltk

nltk.download('stopwords')
```

Finally, run the model from the command line:

```
python3 lda_analyzer.py
```

## Results

The model will return a dictionary of topic vectors for each medical condition in the chief complaints section of the 
records. Each key of the dictionary is a condition and the corresponding value is a list of the top 10 phrases 
associated with that condition.

Two separate models and their results dictionaries have been saved: Model A and Model B. While Model A has a higher 
C_V coherence of 0.69 (versus 0.42 for Model B), the topic vectors for each condition are more general. And while Model
B has a lower coherence, the topic vectors are more specific to the condition and therefore more informative. For this
dataset, a higher number of topics generally yields more overlap between topics and more generic phrases.

Model A can be loaded from `data/model_A.atmodel` and its results from `data/results_A.json`. It was generated using the 
number of topics, k = 145, alpha = 0.61 and beta = 0.61. Additionally, the during preprocessing, only tokens contained 
in at least 2 documents, but no more than 25% of the total documents were kept in the corpus.

Model B and can be loaded from `data/model_B.atmodel` and its results from `data/results_B.json`. It was generated using 
k = 50, alpha = 0.01 and beta = 0.01. During preprocessing, only tokens contained in at least 2 documents, but no more 
than 40% of the total documents were kept in the corpus.

Both models and their results can be loaded as in the following example:

```
import json
from gensim.models import AuthorTopicModel

model_a = AuthorTopicModel.load('data/model_A.atmodel')
with open('data/results_A.json') as f:
    results_a = json.load(f)
```

## Discussion

Extracting factors related to a particular condition from medical text is tricky because much of the text will not be 
related to the condition of interest. For example, screenings sections, social history, family history, etc may include 
some phrases related to the condition, but will also include many false positives. Additionally, medical records often
include many instances of negation (for example, "the patient denies X"). If these negated phrases are not removed 
first, they can appear as false positives as well.

One solution would be to segment the medical record into sections and focus on a subset of those sections (for example, 
just the history of present illness section.) However, this text will be much shorter and likely only represents a 
single topic. For both of these reasons, LDA is no longer a suitable approach since LDA depends on longer documents and 
assumes a distribution of topics in each document. A short text topic modelling (STTM) approach would therefore be 
better. However, parsing each medical record into relevant sections is a challenging task on its own and the STTM 
approach depends on it being done accurately.

An ATM approach mitigates the issue of false positives in family history and other sections because each "author" or 
condition (rather than each document) is associated with a distribution of topics. This means the topic distributions 
are more focused on a single condition, so the similarities between documents with the same condition will be 
highlighted and false positives will carry less weight.

It's clear however that the current approach isn't perfect, since both Model A and Model B return many general terms and 
false positives in their results dictionaries. This could be solved by a clinician reviewing the results and applying 
their domain knowledge to guide the analysis and selection of model hyperparameters. For example, a clinician could 
expand the list of stopwords to include a more comprehensive list of generic clinical terms that should be excluded.

Finally, since an ATM is an extension of LDA, it still requires a large dataset to work effectively. A bigger set 
of records would help the model to better distinguish topics. Additionally, a clinician could aggregate the conditions 
used as authors to build a more concise list of conditions, which would further expand the number of documents 
associated with each condition.
