# NLP
### NLP Project I Worked on During My Degree.

During our work, we wanted to explore the differences between the songs of our time (the 21st century) and older songs written by poets of the "Generation of the State"

# Research Question
Does the content of Hebrew songs written between 1930-1970 (by the poets of the Generation of the State) differ from that of contemporary songs (songs written in the 21st century)? If so, what are the differences, and what do they tell us about the changing ethos of the Jewish people?

As part of the project, I read and implemented the paper:
### Stylometry-based Approach for Detecting Writing Style Changes in Literary Texts

The paper proposes a method for detecting changes in writing style using stylometric features.

# Implementation of the Methods Presented in the Paper on Our Data
The data consists of two groups of songs - old songs and new songs. Using the methods presented in the paper, we tried to find out two things:

### Can we use machine learning tools to distinguish between an old and a new song?
### What is the difference between new and old songs?

### To answer these questions, we divided the features into three categories, as suggested in the paper:

Phraseology analysis
Punctuation analysis
Lexical usage analysis
Since our data is Hebrew songs, we used Hebrew stop words from the nltk API in lexical usage analysis.

Based on these features, we created four types of vectors (one vector for each category and one vector for all categories) that we trained using scikit-learn.

In scikit-learn, we used three methods:

Logistic Regression
SVM Liblinear
SVM Libsvm



In addition, in order to examine the effectiveness of the features taken in the paper, we performed an additional analysis of the data using a feature set from Malu (The Academic Center for Examinations and Evaluation).
The Malu feature set includes 41 features. Some of the features are similar to the features taken in the paper (number of words, word length, etc.), and some are specific to the Hebrew language that the paper does not take into account (percentage of verbs in Hebrew, etc.).
