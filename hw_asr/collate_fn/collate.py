import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    # TODO: your code here
    # keys = ['audio', 'spectrogram', 'text_encoded', 'duration', 'text', 'audio_path']
    # new_keys = ['audio', 'spectrogram', 'text_encoded', 'duration', 'text', 'audio_path',
    #             'text_encoded_length', 'spectrogram_length']

    for key in ['duration', 'text', 'audio_path']:
        result_batch[key] = [item[key] for item in dataset_items]
    
    for key in ['audio', 'text_encoded']:
        result_batch[key] = nn.utils.rnn.pad_sequence([item[key].squeeze() for item in dataset_items], 
                                                      batch_first=True)
        
    spectograms = [item['spectrogram'].squeeze().permute(1, 0) for item in dataset_items]
    result_batch['spectrogram'] = nn.utils.rnn.pad_sequence(spectograms, batch_first=True).permute(0, 2, 1)
    
    for key in ['text_encoded', 'spectrogram']:
        result_batch[key+'_length'] = torch.tensor([value.size(-1) for item in dataset_items for k, value in item.items() if k == key])
   
    return result_batch
