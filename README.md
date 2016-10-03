================
``jobdescription2jobtitle`` readme
================

Introduction
------------

This program given a piece of text such as a cv, job summary or
a Linkdein profile converts it to a 300d vector (using average of word vectors)
and ranks ONET job titles based on similarity to that description.
The ONET is a standard dataset consisting of about 1100 job titles
and their description. It includes other information about jobs that
we didn't use here.

For each job title and description, a 300d average word vector is built.
Given a piece of text the program finds the most similar job titles related
to that text.

The similarity/distance distribution of a piece of text to a 1100d job titles
can be used for comparison to another piece of text to see if both pieces of
text are corresponding to one person or not using cosine distance between them.

If two pieces of text correspond to the same person their distance to 1100 job
titles should be similar (their cosine distance should be low).

The cosine distance between two pieces of text can be used as a single feature
when trying to decide if two pieces of text correspond to a single person or not.

To run the program gensim should be installed and the pre-trained Google word2vec
file should be downloaded and the path in the source changed accordingly.

Pre-trained word vectors
------------------------
download them from https://docs.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download.


Job Title and Description
-------------------------
can be downloaded from ONET dataset here https://www.onetcenter.org/dl_files/database/db_21_0_text/Occupation%20Data.txt.




Contact
-------
Afshin Rahimi <afshinrahimi@gmail.com>
