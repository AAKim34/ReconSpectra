import lightning as L
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanSquaredError
from torchmetrics.regression import MinkowskiDistance
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler

# Lightning
class Pipeline(L.LightningModule):


    def __init__(
        self,
        model,
        exp_name="baseline",
        criterion=nn.MSELoss(),
        num_classes=10,
        optimizer_class=torch.optim.SGD,
        optimizer_kwargs={"lr": 0.01},
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        metrics = MetricCollection([MeanSquaredError(),MinkowskiDistance(2)])
        self.train_metrics = metrics.clone(postfix="/train")
        self.valid_metrics = metrics.clone(postfix="/val")
        self.test_metrics = metrics.clone(postfix="/test")

        # Additionally, we will save training logs “manually”
        # for visualization within the lecture. Please limit yourself
        # to the native training logging tools from PytorchLightning
        # when training your own models.
        self.history = {
            "Loss/train": [],
            "Loss/val": [],
            "MeanSquaredError/train": [],
            "MeanSquaredError/val": [],
            "MinkowskiDistance/train": [],
            "MinkowskiDistance/val": [],
            "model_name": exp_name,
        }

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.model.parameters(), **self.optimizer_kwargs
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)

        self.log("Loss/train", loss, prog_bar=True)
        self.train_metrics.update(out, y)

        # aux logging
        self.history["Loss/train"].append(loss.cpu().detach().item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)

        self.log("Loss/val", loss, prog_bar=True)
        self.valid_metrics.update(out, y)

        # aux logging
        self.history["Loss/val"].append(loss.cpu().detach().item())

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)

        self.test_metrics.update(out, y)


    def on_training_epoch_end(self):
        train_metrics = self.train_metrics.compute()
        self.log_dict(train_metrics)
        self.train_metrics.reset()

        # aux logging
        for key, value in train_metrics.items():
            self.history[key].append(value)

    def on_validation_epoch_end(self):
        valid_metrics = self.valid_metrics.compute()
        self.log_dict(valid_metrics)
        self.valid_metrics.reset()

        # aux logging
        for key, value in valid_metrics.items():
            self.history[key].append(value)

    def on_test_epoch_end(self):
        test_metrics = self.test_metrics.compute()
        self.log_dict(test_metrics)
        self.test_metrics.reset()



# датасет для MLP
class Spec_PDD(Dataset): 
    def __init__(self, data, label, transform = None):
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        label_idx = torch.tensor(self.label[idx])
        data_idx = torch.tensor(self.data[idx])

        if self.transform:
          data_idx = self.transform(data_idx)


        return data_idx.to(torch.float32), label_idx.to(torch.float32)
    


# Архитектруа MLP
class RecSpec(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(125, 3000)  
        # self.fc2 = nn.Linear(250, 500)
        # self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(3000, 190)

        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.5)
        self.dp3 = nn.Dropout(0.2)


    def forward(self, x): 
        x = self.activation(self.fc1(x))
        # x = self.activation(self.fc2(x))
        # x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

# Датасет для RESNET
class Spec_PDD_res(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        label_idx = torch.tensor(self.label[idx])
        data_idx = torch.unsqueeze(torch.tensor(self.data[idx]), 0)

        return data_idx.to(torch.float32), label_idx.to(torch.float32)

# Добавление шума 
class GNoise:

    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, spec):
        cooff = np.where(spec < 0.005, 0, 1)

        spec_noise = abs(spec.numpy() + np.random.normal(0, cooff*self.sigma))
        return torch.from_numpy(spec_noise)
    

class Norm_cust(StandardScaler):

    def __call__(self, spec):
        
        if len(spec.shape)==1:
            spec = spec.reshape(1,-1)
        
        ans = super().transform(spec.numpy())

        return torch.from_numpy(ans)[0]