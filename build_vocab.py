import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO
from tqdm.auto import tqdm


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {} # dict, word is key and index is value
        self.idx2word = {} # dict, idx is key and word is value
        self.idx = 0

    def add_word(self, word):
        # 기존에 word2idx 목록에 없었던 새로운 word를 만나면
        # idx(index) 값을 부여하고, idx를 1 증가시킴
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        # Class를 호출하면 word의 idx값을 반환
        # word2idx에 등록되지 않은 word에 대해서 <unk> token의 index를 반환
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption']) # json file의 caption 부분을 불러옴
        tokens = nltk.tokenize.word_tokenize(caption.lower()) # 불러온 caption의 lower case를 토큰화
        counter.update(tokens)

    # Counter module은 각 token의 등장 횟수를 세는 역할인듯
    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    # threshold 이상 등장한 word만 최종적으로 vocab에 추가
 
    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary() # 앞서 정의한 Vocabulary classd의 instance 생성 
    vocab.add_word('<pad>') # Padding token
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(args):
    nltk.download('punkt')
    vocab = build_vocab(json=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f) # vocab를 f에 저장
    print("Total vocabulary size: {}".format(len(vocab))) # __len__()이 호출됨
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='data/annotations/captions_train2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
