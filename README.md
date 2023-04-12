# compounds-analysis-bert
The code for the paper "A Psycholinguistic Analysis of BERT’s Representations of Compounds"

## Data

The compound dataset is provided. 

For the GloVe vectors the following dataset was used and should be placed in the ```data/``` folder: https://nlp.stanford.edu/data/glove.6B.zip . 

The cleaned wikipedia that is used for the contextual vectors can be found at: https://www.lateral.io/resources-blog/the-unknown-perils-of-mining-wikipedia

## Usage

To run all experiments as seen in the paper run
```python main.py```

There are three different optional arguments. See ```python main.py --help```

The code to make the plots is available in plots.ipynb


## Under construction
- Contextual Vectors
