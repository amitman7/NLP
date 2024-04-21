# NLP
NLP Project I Worked on During My Degree.  
During our work, we wanted to explore the differences between the songs of our time (the 21st century) and older songs written by poets of the "Generation of the State" (Dor HaMedina)

# Research Question
Does the content of Hebrew songs written between 1930-1970 (by the poets of the Generation of the State) differ from that of contemporary songs (songs written in the 21st century)? If so, what are the differences, and what do they tell us about the changing ethos of the Jewish people?

As part of the project, I read and implemented the paper:
Stylometry-based Approach for Detecting Writing Style Changes in Literary Texts

The paper proposes a method for detecting changes in writing style using stylometric features.

# Implementation of the Methods Presented in the Paper on Our Data
The data consists of two groups of songs - old songs and new songs. Using the methods presented in the paper, we tried to find out two things:

### 1. Can we use machine learning tools to distinguish between an old and a new song?
### 2. What is the difference between new and old songs?

### To answer these questions, we divided the features into three categories, as suggested in the paper:
*"Stylometry-based Approach for Detecting Writing Style Changes in Literary Texts"*

    Phraseology analysis
    Punctuation analysis
    Lexical usage analysis
Since our data is Hebrew songs, we used Hebrew stop words from the nltk API in lexical usage analysis.

Based on these features, we created four types of vectors (one vector for each category and one vector for all categories) that we trained using scikit-learn.

In scikit-learn, we used three methods:

    Logistic Regression
    SVM Liblinear
    SVM Libsvm

* we used the code suggested in the paper from: *github.com/jpotts18/stylometry* and made some changes to adapt it to our data and goals.

# results

The results show that it is possible to predict whether a song is old or new with high accuracy (75-85%) using all methods.

Additionally, on our data, the most effective features using Logistic Regression were lexical usage and a combination of all features, with an accuracy of 78%.

Using SVM liblinear, the most effective feature was lexical usage, with an accuracy of 89%.

Using SVM libsvm, the most effective features were phraseology and a combination of all features, with an accuracy of 94%.

# Malu features
In addition, in order to examine the effectiveness of the features taken in the paper, we performed an additional analysis of the data using a feature set from Malu (National Institute for Examinations and Evaluation).

The Malu feature set includes 41 features. Some of the features are similar to the features taken in the paper (number of words, word length, etc.), and some are specific to the Hebrew language that the paper does not take into account (percentage of verbs in Hebrew, etc.).

# results of the Malu features
The results show that the new feature set achieves similar high accuracy to the feature set in the paper. In some cases, the new feature set even performs better (e.g., Logistic Regression).

We hypothesize that the high accuracy is due to the careful selection of features in the new feature set. However, we cannot definitively conclude that the specific selection of features relevant to Hebrew texts improves the results.

# conclusion first article
### These results suggest that there is no single most effective type of feature for all songs, and that the best feature set depends on the method used. 
![Figure_1](https://github.com/amitman7/NLP-/assets/118345516/a71e8861-c95a-432e-838b-e343e45f3b8b)

it should be noted  that we performed the model on a relatively limited dataset. It is likely that if we perform it on a larger dataset, we may be able to find an ideal method. This is a question for further research

#  BERT 

To further investigate the differences between the two sets of songs, *"Dor HaMedina"* and *contemporary songs*, we conducted an additional analysis to determine whether there is a correlation between the syntactic structure of the words and the type of song.

This analysis was performed using a morphological tool called *"New Dictate Parser - BERT"*. We parsed each song using BERT and obtained a JSON file containing various analyses for each word in the song. For each word, we extracted its syntactic structure (verb, noun, etc.). This created a new text for each song containing the syntactic word types.

For example, a new text might look like this: "ADV AUX PRON ADJ PRON PRON ADJ ADV NOUN VERB PRON PROPN SCONJ AUX PRON VERB PRON NOUN VERB VERB PRON ADV ADV ADP SCONJ PRON VERB VERB VERB SCONJ NOUN VERB ADP SCONJ PRON VERB VERB"

We then trained the resulting texts using the same process as the previous text, with some modifications to the features. To isolate the effect of syntax, we removed the following feature types from the three types of features:

Phraseology: Features related to different word, sentence, and text lengths. These are not relevant for this analysis because punctuation is used to classify different types of punctuation marks, and the new texts we created do not contain punctuation marks, as they are represented by the word "PUNCT".
We made the corresponding changes in the extract.py file and used the stylometry library to extract the features for training the model.

In essence, the features we will train on are of the type lexical usage, and we will examine the occurrence of different connecting words between the two types of songs.

# results 

Based on the results, it can be inferred that there is a difference in the syntactic structure of words between "Dor HaMedina" songs and contemporary songs, as the different models were able to train with relatively high accuracy.

A future task would be to investigate which types of syntactic words contribute to the differences between the two periods.

 # PCA

In addition to the previous analysis, we conducted a further investigation of the data using k-means clustering and PCA (Principal Component Analysis) to determine whether these algorithms could effectively classify the differences between the two time periods.

![Figure_1](https://github.com/amitman7/NLP/assets/118345516/71e5c6ce-ff70-45f9-b928-90ef9013b81c)

As evident from the results, this approach effectively highlights a significant difference between the two time periods and demonstrates the algorithm's ability to classify the distinct eras.

* for the PCA we used the code of *github.com/jpotts18/stylometry* as well.

# statistics 

To investigate the syntactic differences between the two sets of songs, we employed a multi-step approach:

Text Parsing: We utilized "New Dictate Parser - BERT" to analyze each song's text, generating a JSON file containing various analyses of the words (morphology, syntax, and entity recognition).

Data Extraction: We extracted relevant data from the generated JSON files, focusing on specific analyses that could reveal syntactic patterns.

Statistical Analysis: Employing the extracted data, we conducted a series of statistical analyses to identify potential differences between the two time periods.

# Conclusions statistics

Our analysis of syntactic features, word lengths, and gender usage revealed intriguing differences between the "Dor HaMedina" and contemporary songs:

### Syntactic Features:

Verbs and Adverbs: "Dor HaMedina" songs exhibited a lower frequency of verbs and adverbs compared to contemporary songs.

Adjectives, Nouns, and Punctuation: In contrast, "Dor HaMedina" songs displayed a higher usage of adjectives, nouns, and a broader range of punctuation marks compared to contemporary songs.

Other Syntactic Word Types: Further investigation revealed additional distinctions in the usage of specific syntactic word types.

### Word Lengths:

Word Length: The average word length in "Dor HaMedina" songs was similar to that in contemporary songs.

Sentence and Song Length: However, the average sentence length and song length were nearly twice as long in contemporary songs compared to "Dor HaMedina" songs.

### Gender Usage:

Gender Distribution: The results indicated no significant difference in gender usage between "Dor HaMedina" and contemporary songs, with a roughly equal distribution of masculine and feminine words.
Overall Summary:

The statistical analysis highlights substantial differences between "Dor HaMedina" and contemporary songs. The most pronounced distinctions lie in the usage of syntactic features and sentence and song lengths. Gender usage, on the other hand, remained relatively consistent across the two eras.

# second article

The paper *"Corpus Periodization Framework to Periodize a Temporally Ordered Text Corpus"* presents a framework called "CorPerds" that enables the automatic classification of text into different time periods. This is achieved by first performing a preliminary temporal classification by the method's implementer, followed by a more refined classification into a smaller and unique set of time periods while preserving the chronological order of the texts.

This paper is unique in that, at the time of its writing, there were no relatively simple and easy-to-implement methods available for this concept, making it a powerful research tool for many parties.

The proposed framework suggests dividing the texts at hand into temporal segments initially by arbitrarily selecting a time unit (hours, days, months, or years) that naturally divides our texts into preliminary segments.

Next, we aim to further consolidate these segments by examining the proximity of pairs of neighboring segments using various methods and making decisions based on whether or not to merge them.

# third article

# Conclusion

In conclusion, the study revealed significant differences between "Dor HaMedina" and contemporary songs. These distinctions manifest in the songs content, syntax, various syntactic elements, and sentence and song lengths. While some of these changes provide insights into the ethos and spirit of the respective eras, others remain difficult to definitively interpret in terms of their temporal context.



