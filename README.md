# image-captioning

## Reference

[image-captioning from pytorch-tutorial by yunjey@github](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)

## Difference

### Preprocessing

- Added nltk.download('punkt') inside build_vocab.py file
  - It'll download punkt file if it doesn't exist
- Applied tqdm module

### Model

- Added 1 more linear layer in Encoder
  - Previous: Linear -> BN
  - Current: Linear -> BN -> ReLU -> Linear -> BN
- Added 1 more linear layer in Decoder
  - Previous: Linear
  - Current: Linear -> ReLU -> Linear
- Replaced LSTM in Deocder to GRU

### Sample

- Fixed about punctuation
  - Previous: A man is running .
  - Current: A man is running.

### Train

- Added RandomVerticalFlip() transform
