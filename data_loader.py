import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB') # 불러온 이미지를 RGB 채널로 변환
        if self.transform is not None:
            image = self.transform(image) # transform이 있으면 적용

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>')) # <start>에 해당하는 token index를 추가
        caption.extend([vocab(token) for token in tokens]) # 각 token의 index 추가
        # 단어를 extend 할 때, vocab(token)에서 Vocabulary()의 __call__() 이 호출되는데, 
        # 이 때 Vocabulary에 등록되지 않은 단어에 대해서는 <unk> 반환
        caption.append(vocab('<end>')) # 문장 끝에 <end> 추가

        target = torch.Tensor(caption)
        return image, target
    
    def __len__(self):
        return len(self.ids)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    # data의 column 0,1 중에 1에 대해서 정렬, x[1](caption)의 길이에 대해 내림차순 정렬
    images, captions = zip(*data) # make iterable tuple

    # torch.stack(): Concatenates a sequence of tensors along a new dimension.
    # https://pytorch.org/docs/stable/generated/torch.stack.html
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long() # (number of captions, maximum size of caption)
    for i, cap in enumerate(captions):
        end = lengths[i] # length of each caption
        targets[i, :end] = cap[:end] # caption 값을 채워넣고, 빈 부분은 0이 됨 -> Padding

    return images, targets, lengths

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root, json=json, vocab=vocab, transform=transform)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, batch_size=batch_size, 
                                              shuffle=shuffle, num_workers=num_workers,
                                              collate_fn=collate_fn)

    return data_loader
