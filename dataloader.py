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


def causal_mean_replacement(features, umask):
    """Replace each token with its own dialogue's valid history/current mean."""
    valid = umask.transpose(0, 1).to(device=features.device, dtype=features.dtype).unsqueeze(-1)
    sums = (features * valid).cumsum(dim=0)
    counts = valid.cumsum(dim=0).clamp_min(1)
    return sums / counts * valid


def validate_checkpoint_protocol(checkpoint, causal=False, expected_args=None):
    """Reject incompatible temporal protocols before loading model/optimizer state."""
    saved_args = checkpoint.get('args', {})
    if not isinstance(saved_args, dict):
        raise ValueError('checkpoint args must be a dictionary')
    saved_causal = saved_args.get('causal', checkpoint.get('causal', False))
    if not isinstance(saved_causal, bool) or saved_causal != causal:
        raise ValueError('checkpoint causal protocol does not match requested causal={}'.format(causal))
    if 'causal' in checkpoint and checkpoint['causal'] != saved_causal:
        raise ValueError('checkpoint causal metadata is inconsistent')
    architecture_keys = (
        'Dataset', 'base_model', 'norm', 'num_L', 'num_K', 'av_using_lstm',
        'use_residue', 'graph_type', 'modals', 'mm_fusion_mthd', 'use_speaker',
        'use_modal', 'use_gcn', 'Deep_GCN_nlayers', 'multi_modal',
    )
    if expected_args is not None:
        expected_args = expected_args if isinstance(expected_args, dict) else vars(expected_args)
        for key in architecture_keys + ('selection_protocol',):
            if key not in expected_args:
                continue
            if key not in saved_args:
                if causal:
                    raise ValueError('causal checkpoint is missing {} metadata'.format(key))
                continue  # Legacy official baseline checkpoints have no args.
            if saved_args[key] != expected_args[key]:
                raise ValueError('checkpoint {}={!r} does not match requested {!r}'.format(
                    key, saved_args[key], expected_args[key]))
    if causal and saved_args.get('norm') != 'LN':
        raise ValueError('causal checkpoint must record effective norm=LN')


def load_curriculum_difficulties(dataset, causal=False):
    suffix = '_causal_diff.pkl' if causal else '_diff.pkl'
    path = 'caldiff/{}{}'.format(dataset, suffix)
    with open(path, 'rb') as handle:
        payload = pickle.load(handle)
    if not causal:
        return payload
    if not isinstance(payload, dict) or payload.get('causal') is not True:
        raise ValueError('causal curriculum requires a causal difficulty wrapper: {}'.format(path))
    if payload.get('dataset') != dataset or not isinstance(payload.get('difficulties'), dict):
        raise ValueError('invalid causal difficulty dataset or values: {}'.format(path))
    return payload['difficulties']


class IEMOCAPDataset(Dataset):

    def __init__(self, train=True, epoch_ratio=-1, keys=None, causal=False):
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
            self.diff = load_curriculum_difficulties('IEMOCAP', causal=causal)
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

    def __init__(self, path, train=True, epoch_ratio=-1, split=None, keys=None, causal=False):
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
            self.diff = load_curriculum_difficulties('MELD', causal=causal)
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
