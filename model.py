import torch
import torch.nn as nn
import torchvision.models as models # Check https://pytorch.org/vision/stable/models.html and find ResNet
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]          # we will not use the last fc layer
        self.resnet = nn.Sequential(*modules)           # our resnet won't include the last fc layer
        middle_size = int(resnet.fc.in_features / 2)    # resnet.fc.in_features = 2048, out_features = resnet에서 분류할 class 갯수
        self.linears = nn.Sequential(nn.Linear(resnet.fc.in_features, middle_size),
                                    nn.BatchNorm1d(middle_size, momentum=0.01),
                                    nn.Linear(middle_size, embed_size),
                                    nn.BatchNorm1d(embed_size, momentum=0.01))
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1) # it'll be 1D torch tensor, torch.Size([embed_size])
        features = self.linears(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linears = nn.Sequential(nn.Linear(hidden_size, int(hidden_size/2)),
                                    nn.Linear(int(hidden_size/2), vocab_size))
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        # unsqueeze: features를 torch.Size([embed_size])에서 torch.Size([embed_size, 1])로 변환
        # cat: 둘을 붙임 (unsqueeze가 이것 때문에 필요함, 차원이 서로 맞아야 하기 때문)
        # image에서 추출된 feature와 caption이 각각 일대일로 대응되는 tensor가 만들어짐, torch.Size([embed_size, 2])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        # https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linears(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        # 여기에서는 caption이 주어지지 않음
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linears(hiddens.squeeze(1))           # outputs:  (batch_size, vocab_size)
            # 1에 해당하는, 즉 2번째 dimension인 vocab 중에서 각 Batch마다 가장 값이 큰 (가장 그럴듯한) 결과를 고름
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            # 고른 값을 결과값 caption에 추가
            sampled_ids.append(predicted)
            # use predicted word as next input for LSTM
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids