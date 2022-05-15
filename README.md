# NLP Task Solver

This is a project for our Intelligent Systems course.

This project was initially about developing a simple question-answering (QA) app based on a pre-trained Transformer encoder. During the development, we became so amazed by the flexibility of Transformer models that we decided to experiment more and created "Natural Language Task Solving App", an app that solves NLP problems like NER, GEC, MCQA using only Transformer-based models.

## Models

We trained / fine-tuned every model for this app by ourselves. We also developed all of the prediction heads. All models used in this app are Transformer-based.

This repository uses Google Drive to store large files. To see all model files, clone this repository and run `application.py`.

  Model | Type | Parameters | Training data | Training time | Evaluation
---|---|---|---|---|---
`question_answering`/`frozen` | Frozen pre-trained Transformer encoder and trained prediction heads | 11M | SQuAD 1.1 | 24 hours on CPU | 32.3 % F1 
`question_answering`/`fine_tuned` | Fine-tuned pre-trained Transformer encoder and trained prediction head | 11M | SQuAD 1.1 | 5 hours on RTX 3090 | 84.1 % F1 
`grammar_correction`/`grammar_corrector` | Fine-tuned pre-trained Transformer encoder-decoder | 60M | Subset of C4_200M | 3 hours on RTX 3080 Ti | _ 
`named-entity-recognition`/`ner_tagger` | Frozen pre-trained Transformer encoder and trained prediction head | 11M | CoNLL-2003 | 6 hours on CPU | _ 
`multiple-choice`/`choice_picker` | Fine-tuned pre-trained Transformer encoder-decoder | 220M | RACE | 3 hours on RTX 3060 | 67.5 % 
`natural_language_inference`/`discrete` | Fine-tuned pre-trained Transformer encoder and trained prediction head. Premise and hypothesis encoded separately | 11M | MultiNLI | 2.5 hours on RTX 3090 | 60.3 %
`natural_language_inference`/`joined` | Fine-tuned pre-trained Transformer encoder and trained prediction head. Premise and hypothesis encoded together | 11M | MultiNLI | 4 hours on RTX 3090 | 81.6 % 


## Usage

This is a web app based on the Flask framework. To open the GUI, run `application.py`, then follow the URL displayed (http://127.0.0.1:5000/ by default).

On the first run, the app will take some time to download large files from Google Drive.

## Question Answering (QA)

Insert a passage and a question related to the passage. The app will answer your question by extracting a piece of text from the passage.

![qa](https://user-images.githubusercontent.com/62071233/148637260-37f6d2e9-4156-49f6-9b49-e8adb627712b.png)

## Multiple Choice Question Answering (MCQA)

Insert a passage and a question related to the passage, followed by 4 possible choices. The app will pick one of your provided choices.  
Example question format: `<question> A. <choice 0> B. <choice 1> C. <choice 2> D. <choice 3>`

![mcqa](https://user-images.githubusercontent.com/62071233/148637306-f142003c-caa6-4df9-a571-2073e41a2cb7.png)

## Named Entity Recognition (NER)

Insert a passage and a request for named entity recognition. The app will return a list of all of the named entities in the passage.  
Example question: `What are the named entities in this passage?`

![ner](https://user-images.githubusercontent.com/62071233/148637283-8b78023f-95ab-4958-934e-6aebc40191af.png)

## Grammatical Error Correction (GEC)

Insert a passage with grammatical errors and a request to fix grammatical errors. The app will return a list of fixed errors in the passage.  
Example question: `Correct this passage?`

![gec](https://user-images.githubusercontent.com/62071233/148637297-dbb4c7b8-d0ff-47f2-94f2-9bbd06e691af.png)


