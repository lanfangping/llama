import pandas as pd
import sys
from os.path import dirname, abspath
root = dirname(dirname(abspath(__file__)))
print(root)
sys.path.append(root)

from typing import List, Optional
from llama import Llama, Dialog

def construct_dialogs_in_batch(path_to_csv_file, batch = 5, begin_index = 0):
    data = pd.read_csv(path_to_csv_file)
    # print(data[0:5])
    i = begin_index
    while i < len(data):
        dialogs: List[Dialog] = []
        pair_ids = []

        if len(data) - i < 5:
            batch = len(data) - i
        
        for j in range(batch):
            old_snippet = data['old_snippet'][i][2:] # remove -/+ at begining
            new_snippet = data['new_snippet'][i][2:]
            pair_ids.append(data['pair_id'][i+j])
            d = [
                    {
                        "role": "system",
                        "content": """
                            Classify the type of revision between the two paragraphs. The paragraph will be delimited with triple quotes.

                            The categories of revisions:

                            1.1: Surface Changes
                                - Abbreviation, Spelling, Grammer, Punctuation
                            1.3: Word and Sentence-level Edits
                                - Rephrase, Paraphrase

                            2.1: Organizational Changes
                                - Refactoring, Relocation

                            3.1: Information Updates
                                - Fact Updates, Clarification, DIsambiguation
                            3.2: Information Addition
                            3.3: Information Deletion
                                - Simplification, Concision

                            4.1: Citations, References, and Linking Changes 
                                - Table/Figure Reference Changes,  Linkings Changes

                            5: Others

                            Lables are [1.1, 1.3, 2.1, 3.1, 3.2, 3.3, 4.1, 5]. Output JSON data that includes multiple matching category indexes into an array. The format of the JSON is {'categories':[...]}.
                            """,
                    },
                    {   "role": "user", 
                        "content": """
                            \"\"\"{}\"\"\"\n\n\"\"\"{}\"\"\"
                            """.format(old_snippet, new_snippet)
                    }
            ]
            dialogs.append(d)

        i += batch 
        yield pair_ids, dialogs

def convert_to_csv(data_list, csv_filename='output.csv'):
    # Convert the list of maps to a DataFrame
    df = pd.DataFrame(data_list)

    # Write the DataFrame to a CSV file
    df.to_csv(csv_filename, index=False)
    print(f"CSV file '{csv_filename}' created successfully.")

if __name__ == '__main__':
    print("llll")
    path_to_csv_file='../../CompareDBDoc/datasets/data_humanlabel/sampled_data_Jan19.csv'
    for pair_ids, dialogs in construct_dialogs_in_batch(path_to_csv_file):
        print("pair_ids", pair_ids)
        print("dialogs", dialogs)
        # exit()

    # path_to_csv_file="../datasets/data_humanlabel/sampled_data_Jan19.csv"
    # construct_pairs_with_id_without_prompt(path_to_csv_file, begin_index=0)