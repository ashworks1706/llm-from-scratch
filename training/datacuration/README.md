data curation is how raw web text becomes pretraining data. most of the quality of a model comes from this step.

we extract text from html, score quality with a small classifier, remove near duplicates with minhash, filter pii and toxic text, and remove benchmark questions from the training set.
