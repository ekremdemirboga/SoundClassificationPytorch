
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from urbansounddataset import UrbanSoundDataset
from cnn import CNNNetwork

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
prefix = "C:/Users/ekrem/Desktop/makamFinder/pytorch_test/audio_test/Data/UrbanSound8K/"
ANNOTATIONS_FILE = prefix + "metadata/UrbanSound8K.csv"
AUDIO_DIR = prefix + "audio"
SAMPLE_RATE = 16000
n=3
NUM_SAMPLES = SAMPLE_RATE*n ## it means n seconds of audio



def train_one_epoch(model,data_loader,loss_fn,optimizer, device):
    for inputs,targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        #calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions,targets)
        
        #backpropagate loss and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"loss: {loss.item()}")
    
def train(model,data_loader,loss_fn,optimizer, device,epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model,data_loader,loss_fn,optimizer, device)
        print("------------------")
    print("Training is done.train_data_loader")



##initiate data set
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"using {device} device")

mel_spectogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

usd = UrbanSoundDataset (ANNOTATIONS_FILE, AUDIO_DIR, mel_spectogram,
                        SAMPLE_RATE,NUM_SAMPLES,device)

train_data_loader = DataLoader(usd,batch_size=BATCH_SIZE)

if __name__ == "__main__":

    #build model
    cnn = CNNNetwork().to(device)
    ## Lost function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(),lr = LEARNING_RATE)

    ##Train Model
    train(cnn,train_data_loader, loss_fn, optimizer, device, EPOCHS)

    torch.save(cnn.state_dict(),"cnn.pth")
    print("model trained and saved")



