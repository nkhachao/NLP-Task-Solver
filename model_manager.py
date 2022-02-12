from pathlib import Path
import os
import gdown
import requests

project_root = str(Path(__file__).parent)

model_directories = ['/models/multiple-choice/choice_picker/pytorch_model.bin',
                     '/models/grammar-correction/grammar_corrector/pytorch_model.bin',
                     '/models/natural_language_inference/discrete/transformer/pytorch_model.bin',
                     '/models/natural_language_inference/joined/transformer/pytorch_model.bin',
                     '/models/question_answering/fine_tuned/transformer/pytorch_model.bin',
                     '/models/question_answering/fine_tuned/prediction_head/span_picker']

model_file_ids = ['1JarYCNGA2-MFCM4yT_gRDI0ahG1lKEcQ',
                  '1hHBy3W9GHshxlk74IcWRwvyWPGRzG_hv',
                  '1qGCFfadU-glx2KFwPd-qV4B0DBy2TRph',
                  '1eYqtm4-yf08wLT_1hagr-_vUBOIz1yF_',
                  '1C2V64e7Ic6P7hRRRSk1BqrqLOpXx9fXV',
                  '1lEHJSb81AXRGza6ZW74f0Tghe94mwgI2']


def download_model(index, save_directory):
    url = 'https://drive.google.com/uc?id='+model_file_ids[index]
    gdown.download(url, save_directory, quiet=False)


def fetch_models():
    for model_path, model_file_id, i in zip(model_directories, model_file_ids, range(len(model_directories))):
        print('Fetching models', i+1, '/', len(model_directories), '  (', model_path, ')')
        if not os.path.isfile(project_root + model_path):
            download_model(i, project_root + model_path)


def clear_model_files():
    for file_path in model_directories:
        if os.path.isfile(project_root + file_path):
            os.remove(project_root + file_path)
            print('1 file deleted')


if __name__ == "__main__":
    clear_model_files()
else:
    fetch_models()