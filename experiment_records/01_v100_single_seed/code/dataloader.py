import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import pickle, pandas as pd
import numpy


def split_iemocap_dialogues(keys, seed, valid_fraction=0.1):
    """Return deterministic, disjoint dialogue-level train/validation keys."""
    keys = list(keys)
    if not 0 <= valid_fraction < 1:
        raise ValueError("valid_fraction must be in [0, 1)")
    validation_size = int(len(keys) * valid_fraction)
    if valid_fraction > 0 and len(keys) > 1:
        validation_size = max(1, validation_size)
    shuffled = list(keys)
    numpy.random.RandomState(seed).shuffle(shuffled)
    validation_set = set(shuffled[:validation_size])
    train_keys = [key for key in keys if key not in validation_set]
    validation_keys = [key for key in keys if key in validation_set]
    return train_keys, validation_keys


def resolve_meld_split(train_ids, valid_ids, test_ids, split):
    """Resolve the official MELD dialogue IDs without validation leakage."""
    valid_set = set(valid_ids)
    if split == 'train':
        return [key for key in train_ids if key not in valid_set]
    if split == 'validation':
        return list(valid_ids)
    if split == 'test':
        return list(test_ids)
    raise ValueError("split must be one of: train, validation, test")


def select_curriculum_keys(keys, difficulty, ratio):
    """Select easiest eligible dialogues, then restore the source ordering."""
    keys = list(keys)
    missing = [key for key in keys if key not in difficulty]
    if missing:
        raise KeyError("difficulty file is missing {} eligible dialogues".format(len(missing)))
    selected_count = int(ratio * len(keys))
    ranked = sorted(keys, key=lambda key: difficulty[key])[:selected_count]
    selected = set(ranked)
    return [key for key in keys if key in selected]


class IEMOCAPDataset(Dataset):

    def __init__(self, train=True, epoch_ratio=-1, keys=None):
        explicit_keys = keys is not None
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('data/IEMOCAP_features.pkl', 'rb'), encoding='latin1')

        _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4,\
        _, _, _, _ = pickle.load(open('data/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.keys = list(keys) if keys is not None else [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

        if epoch_ratio != -1:
            assert epoch_ratio > 0 and epoch_ratio <= 1
            self.diff = pickle.load(open('caldiff/IEMOCAP_diff.pkl', 'rb'))
            if explicit_keys:
                self.keys = select_curriculum_keys(self.keys, self.diff, epoch_ratio)
            else:
                sorted_diff = [item[0] for item in sorted(self.diff.items(), key=lambda x: x[1])]
                sorted_diff = sorted_diff[:int(epoch_ratio*len(sorted_diff))]
                keysdict = {item:item_index for item_index, item in enumerate(self.keys)}
                sorted_diff = {item:keysdict[item] for item in sorted_diff}
                self.keys = [item[0] for item in sorted(sorted_diff.items(), key=lambda x: x[1])]
            self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(numpy.array(self.roberta1[vid])),\
               torch.FloatTensor(numpy.array(self.roberta2[vid])),\
               torch.FloatTensor(numpy.array(self.roberta3[vid])),\
               torch.FloatTensor(numpy.array(self.roberta4[vid])),\
               torch.FloatTensor(numpy.array(self.videoVisual[vid])),\
               torch.FloatTensor(numpy.array(self.videoAudio[vid])),\
               torch.FloatTensor(numpy.array([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]])),\
               torch.FloatTensor(numpy.array([1]*len(self.videoLabels[vid]))),\
               torch.LongTensor(numpy.array(self.videoLabels[vid])),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i], True) if i<9 else dat[i].tolist() for i in dat]

class MELDDataset(Dataset):

    def __init__(self, path, train=True, epoch_ratio=-1, split=None, keys=None):
        explicit_split = keys is not None or split is not None
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, _ = pickle.load(open(path, 'rb'))

        _, _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
            _, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open("data/meld_features_roberta.pkl", 'rb'), encoding='latin1')

        if keys is not None:
            self.keys = list(keys)
        elif split is not None:
            self.keys = resolve_meld_split(self.trainIds, self.validIds, self.testIds, split)
        else:
            # Backwards-compatible official behavior.
            self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)
        if epoch_ratio != -1:
            assert epoch_ratio > 0 and epoch_ratio <= 1
            self.diff = pickle.load(open('caldiff/MELD_diff.pkl', 'rb'))
            if explicit_split:
                self.keys = select_curriculum_keys(self.keys, self.diff, epoch_ratio)
            else:
                sorted_diff = [item[0] for item in sorted(self.diff.items(), key=lambda x: x[1])]
                sorted_diff = sorted_diff[:int(epoch_ratio*len(sorted_diff))]
                keysdict = {item:item_index for item_index, item in enumerate(self.keys)}
                sorted_diff = {item:keysdict[item] for item in sorted_diff}
                self.keys = [item[0] for item in sorted(sorted_diff.items(), key=lambda x: x[1])]
            self.len = len(self.keys)


    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(numpy.array(self.roberta1[vid])),\
               torch.FloatTensor(numpy.array(self.roberta2[vid])),\
               torch.FloatTensor(numpy.array(self.roberta3[vid])),\
               torch.FloatTensor(numpy.array(self.roberta4[vid])),\
               torch.FloatTensor(numpy.array(self.videoVisual[vid])),\
               torch.FloatTensor(numpy.array(self.videoAudio[vid])),\
               torch.FloatTensor(numpy.array(self.videoSpeakers[vid])),\
               torch.FloatTensor(numpy.array([1]*len(self.videoLabels[vid]))),\
               torch.LongTensor(numpy.array(self.videoLabels[vid])),\
               vid  

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i], True) if i<9 else dat[i].tolist() for i in dat]
