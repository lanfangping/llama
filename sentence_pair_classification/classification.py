# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional
import fire

from llama import Llama, Dialog
import sys
from os.path import dirname, abspath
root = dirname(dirname(dirname(abspath(__file__)))) # directory to CompareDBDoc
sys.path.append(root) 

from sentence_pair_classification.data import construct_dialogs_in_batch, convert_to_csv
from tqdm import tqdm

import logging
import time
import datetime
current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
log_filename = "./logs/running{}.log".format(current_time)
logging.basicConfig(format='%(asctime)s - %(module)s - %(message)s', filename=log_filename, filemode='w', level=logging.INFO)

path_to_csv_file='../../CompareDBDoc/datasets/data_humanlabel/sampled_data_Jan19.csv'
path_to_output_file = './llama2_7b_data/llama2_7b_response_data_Jan19.csv'
begin_index = 0
batch_size = 5

# for old, new in data.construct_pairs_sequence(path_to_csv_file):
#     print(old)
#     print("=========="*5)
#     print(new)
#     exit()


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    logging.info('Begin to run...')
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    labeled_datas = []
    exceptional_pair_ids = []
    try:
        for pair_ids, dialogs in tqdm(construct_dialogs_in_batch(path_to_csv_file, batch=batch_size, begin_index=begin_index), total=(129-begin_index)//batch_size):
            
            try:
                results = generator.chat_completion(
                    dialogs,  # type: ignore
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )
            except Exception as e:
                exceptional_pair_ids += pair_ids
                logging.error("Exception occurred", exc_info=True)
                logging.info('Exception pair_ids: {}'.format(pair_ids))
                continue

            else:
                for idx, result in enumerate(results):

                    record = {
                        'pair_id': pair_ids[idx],
                        "response": result['generation']['content'].replace("|", "\\")
                    }

                    labeled_datas.append(record)
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
    finally:
        logging.info('Exceptional pair_ids: {}'.format(exceptional_pair_ids))
        logging.info('Finish to run to label data...')
        logging.info('Save labeled data...')
        try:
            filename = ".".join(path_to_output_file.split('.')[:-1])
            extension = path_to_output_file.split('.')[-1].strip()
            # print("filename", filename)
            # print("extension", extension)
            outputname = filename + "_idx" + str(begin_index) + "_" + str(current_time)  + "." + extension
            print(outputname)
            convert_to_csv(labeled_datas, outputname)
        except Exception as e:
            logging.error("Save Exception occurred", exc_info=True)
        else: 
            logging.info('Successfully save labeled data!')

if __name__ == "__main__":
    fire.Fire(main)
    # ckpt_dir = "../llama-2-7b-chat"
    # tokenizer_path = "../tokenizer.model"
    # max_seq_len = 4096 
    # max_batch_size = 6
    # main(ckpt_dir, tokenizer_path, max_seq_len=max_seq_len, max_batch_size=max_batch_size)
