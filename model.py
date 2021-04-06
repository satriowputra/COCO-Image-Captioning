import torch
import torch.nn as nn
# from efficientnet_pytorch import EfficientNet
import torchvision.models as models

# class EncoderCNN(nn.Module):
#     def __init__(self, embed_size):
#         super(EncoderCNN, self).__init__()
#         effnet = EfficientNet.from_pretrained('efficientnet-b0')
#         for param in effnet.parameters():
#             param.requires_grad_(False)
        
#         # modules = list(effnet.children())[:-1]
#         self.effnet = effnet
#         self.effnet._fc = nn.Linear(self.effnet._fc.in_features, embed_size)
#         self.batch_norm = nn.BatchNorm1d(embed_size, momentum=0.01)

#     def forward(self, images):
#         features = self.effnet(images)
#         # features = features.view(features.size(0), -1)
#         # features = self.embed(features)
#         features = self.batch_norm(features)
#         return features
    
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        # modules = list(effnet.children())[:-1]
        self.resnet = resnet
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
        self.batch_norm = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        # features = features.view(features.size(0), -1)
        # features = self.embed(features)
        features = self.batch_norm(features)
        return features
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super(DecoderRNN, self).__init__()
        
        # Set class variables
        self.dropout = 0.4
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define model layers
        self.embeding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=self.dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        # Finding batch size
        batch_size = captions.size(0)
        
        # Init hidden
        # self.hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
        #               torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device))
        
        # Prepare data for lstm
        # features = features.view(batch_size, 1, features.size(1))
        features = features.unsqueeze(1)
        # print("features shape:", features.shape)
        embedding_output = self.embeding(captions[:,:-1])
        # print("embedding_output shape:", embedding_output.shape)
        lstm_in = torch.cat((features, embedding_output), 1) # Concatenate features with embedding output
        
        # lstm
        lstm_out, _ = self.lstm(lstm_in)
        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)
        lstm_out = lstm_out.squeeze(1)
        out = self.fc(lstm_out)
        out = out.view(batch_size, -1, self.vocab_size)
        # print("out shape:", out.shape)
        
        return out

    def sample(self, inputs, states=None, max_len=40):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sample_out = []
        # features = inputs.view(batch_size, 1, features.size(1))
        
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            lstm_out = lstm_out.squeeze(1)
            out = self.fc(lstm_out)
            _, pred = out.max(1)
            sample_out.append(pred.item())
            inputs = self.embeding(pred)
            inputs = inputs.unsqueeze(1)
        return sample_out