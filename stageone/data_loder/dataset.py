# 数据集类
import os

import pandas as pd
from torch.utils.data import Dataset
from typing import List, Tuple


class PromptedRelationExtractionDataset(Dataset):
    def __init__(
            self,
            source_texts: List[str],
            re_labels: List[str]
    ):
        assert len(source_texts) == len(re_labels)
        self.source_texts = source_texts
        self.re_labels = re_labels
    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        item = {'source_texts': self.source_texts[idx],
                're_labels': self.re_labels[idx]}
        return item



def make_relation_extract_dataset(base_path,dataset
        ) -> Tuple[PromptedRelationExtractionDataset]:
    data_dict = {}
    for split in ['train', 'dev', 'test']:
        source_texts, re_labels = \
            load_relation_extract_dataset(dataset,
                                          split, base_path)
        re_dataset = PromptedRelationExtractionDataset(source_texts,
                                                       re_labels)
        data_dict[split] = re_dataset

    return (data_dict['train'], data_dict['dev'], data_dict['test'])


def load_relation_extract_dataset(
        dataset: str,
        split: str,
        base_path: str,
) -> Tuple[List[str]]:
    # 三元组/五元组
    assert dataset in ['triple', 'quintuple']
    assert split in ['train', 'dev', 'test']

    filepath = f'{dataset}/{split}.tsv'
    full_filepath = os.path.join(base_path, filepath)
    # print(os.getcwd())  # 打印当前工作目录
    df = pd.read_csv(full_filepath, sep='\t')
    source_texts = df.text.tolist()
    re_labels = df.label.tolist()
    return (source_texts, re_labels)