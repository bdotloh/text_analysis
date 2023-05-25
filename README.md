# text analysis

A simple workflow for processing times. Combines a series of sequential transformations to a set of raw, unprocessed documents.

The order of transformations, as of 25/05/2023:

1) Preprocess: remove contractions, fix encoding issues.
2) Obtain Document-Term matrix with user specified n-gram
3) Embed text with [Sentence-Transformers](https://www.sbert.net)
4) Reduce embedding dimensions with [UMAP](https://umap-learn.readthedocs.io/en/latest/)
5) Cluster reduced dimensions embeddings with [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html)


