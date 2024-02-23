import os
import torch
from torch import nn, optim, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L


# Define any number of nn.Module (or use your current ones)
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

# Define the LightningModule 
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def training_step(self, batch, batch_idx):
        # trainging step define the train loop
        # It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
# Init the autoencoder
autoencoder = LitAutoEncoder(encoder=encoder, decoder=decoder)

# Define a dataset (Lightning supports ANY iterable (DataLoader, numpy, etc…) for the train/val/test/predict splits.)
dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset)

# Train the model (hint: here are some helpful Trainer argument for rapid idea iteration.)
# trainer = L.Trainer(limit_predict_batches=100, max_epochs=1)
# trainer.fit(model=autoencoder, train_dataloaders=train_loader)

# Use the model
# Load checkpoint
checkpoint = './lightning_logs/version_0/checkpoints/epoch=0-step=60000.ckpt'
autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint_path=checkpoint, encoder= encoder, decoder=decoder)

# Chosse your train nn.Module
encoder = autoencoder.encoder
encoder.eval()

# Embed 4 fake images!
fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
embedding = encoder(fake_image_batch)
print("⚡" * 20, "\nPredictions (4 image embeddings) :\n", embedding, "\n", "⚡" * 20)

