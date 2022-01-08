# Transformer-based Natural Language Task Solving App

This is a project for our Intelligent Systems course.

This project was initially about developing a simple question-answering (QA) app based on a pre-trained Transformer encoder. During the development, we became so amazed by the flexibility of Transformer models that we decided to experiment more and created "Natural Language Task Solving App", an app that solves NLP problems like NER, GEC, MCQA using Transformer-based models.

## Models

We trained / fine-tuned every model for this app by ourselves. We also developed all of the prediction heads. All models used in this app are Transformer-based.

This repository uses Google Drive to store large files. To see all model files, clone this repository and run `application.py`.

  Model | Type | Parameters | Training data | Training time | Evaluation
---|---|---|---|---|---
`question_answering`/`start_probability` | Frozen pre-trained Transformer encoder and trained prediction head | 11M | SQuAD 2.0 | 12 hours on CPU | _ 
`question_answering`/`end_probability` | Frozen pre-trained Transformer encoder and trained prediction head | 11M | SQuAD 2.0 | 12 hours on CPU | _ 
`question_answering`/`classifier` | Frozen pre-trained Transformer encoder and trained prediction head | 11M | SQuAD 2.0 | 12 hours on CPU | _ 
`grammar_correction`/`grammar_corrector` | Fine-tuned pre-trained Transformer encoder-decoder | 60M | Subset of C4_200M | 3 hours on RTX 3080 Ti | _ 
`named-entity-recognition`/`ner_tagger` | Frozen pre-trained Transformer encoder and trained prediction head | 11M | CoNLL-2003 | 6 hours on CPU | _ 
`multiple-choice`/`choice_picker` | Fine-tuned pre-trained Transformer encoder-decoder | 220M | RACE | 3 hours on RTX 3060 | 67.5 % 

## Usage

This is a web app based on the Flask framework. To open the GUI, run `application.py`, then follow the URL displayed (http://127.0.0.1:5000/ by default).

On the first run, the app will take some time to download large files from Google Drive.

## Question Answering (QA)

Insert a passage and a question related to the passage. The app will answer your question by extracting a piece of text from the passage.

![qa](https://user-images.githubusercontent.com/62071233/148634598-8ee8b9b0-a861-4ca9-ab91-013e3c100b04.png)

## Multiple Choice Question Answering (MCQA)

Insert a passage and a question related to the passage, followed by 4 possible choices. The app will pick one of your provided choices.  
Example question format: `<question> A. <choice 0> B. <choice 1> C. <choice 2> D. <choice 3>`

![mcqa](https://user-images.githubusercontent.com/62071233/148634747-e1f0e1c0-e7be-43e2-a56f-0f5a198993c2.png)

## Named Entity Recognition (NER)

Insert a passage and a request for named entity recognition. The app will return a list of all of the named entities in the passage.  
Example question: `What are the named entities in this passage?`

![ner](https://user-images.githubusercontent.com/62071233/148634814-69402548-ef92-4d7a-bb59-c0e67db76a16.png)

## Grammatical Error Correction (GEC)

Insert a passage with grammatical errors and a request to fix grammatical errors. The app will return a list of fixed errors in the passage.  
Example question: `Correct this passage?`

![gec](https://user-images.githubusercontent.com/62071233/148634872-0b3bfcab-dea3-4ea1-8b64-16f952139add.png)

