# Transformer-based Natural Language Task Solving App

This is a project for our Intelligent Systems course.

This project was initially about developing a simple question-answering (QA) app based on a pre-trained Transformer encoder. During the development, we became so amazed by the flexibility of Transformer models that we decided to experiment more and created "Natural Language Task Solving App", an app that solves NLP problems like NER, GEC, MCQA, ... using Transformer-based models.

## Models

We trained / fine-tuned every model for this app by ourselves. We also developed all of the prediction heads. All models developed for this app are Tranformer-based.

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
