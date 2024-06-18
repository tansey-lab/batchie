import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from scipy.special import logit
from batchie.common import (
    ArrayType,
    FloatingPointType,
    CONTROL_SENTINEL_VALUE,
    copy_array_with_control_treatments_set_to_zero,
)
from batchie.core import (
    BayesianModel,
    MCMCModel,
    Theta,
)
from batchie.data import ScreenBase, create_single_treatment_effect_map, ExperimentSpace
from batchie.fast_mvn import sample_mvn_from_precision
logger = logging.getLogger(__name__)

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.utils.data.dataset import random_split
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.init as init



class linear_nn(nn.Module): 
  def __init__(self):
    super(linear_nn, self).__init__()
    self.model = nn.Sequential(
        nn.Linear(5, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )

  def forward(self, x):
    return self.model(x)
  
  def reset_model(self):
    for module in self.model.modules():
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)          

def train_model(model,X_tensor,y_tensor):
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        model = model.cuda()
        logger.info("Using GPU")
    else:
        logger.info("Not using GPU")

    if X_tensor is None:
        raise ValueError('X Tensor is Empty: ' + str(X_tensor))
    if y_tensor is None:
        raise ValueError('Y Tensor is Empty: ' + str(y_tensor))

    logger.info("X_tensor Shape: " + str(X_tensor.shape))
    logger.info("Y_tensor Shape: " + str(y_tensor.shape))

    # Create a TensorDataset
    dataset = TensorDataset(X_tensor, y_tensor)
    # Create a DataLoader
    batch_size = 64
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # print()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training Size: {train_size}, Validation size: {val_size}, Testing Size: {test_size}")
    PRINT_EVERY = 100
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay= 1e-6)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Initialize a list to store MSE values per epoch
    modelname_to_print = 'testing'

    train_losses = []
    val_losses = []
    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        model.train()

        for i, (inputs, targets) in enumerate(train_dataloader,0):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            average_loss = running_loss / train_size
            train_losses.append(average_loss)

            model.eval()  # Set the model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():  # Disable gradient calculation during validation
                for inputs, targets in val_dataloader:
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets.unsqueeze(1)).item() * inputs.size(0)
            val_loss /= len(val_dataloader.dataset)
            val_losses.append(val_loss)
            print(f'Epoch: {epoch}, Validation loss: %.3f' % val_loss)
            scheduler.step(val_loss)
        print('Finished Training')
    plot_training_loss_validation(train_losses,val_losses,modelname_to_print)
    torch.save(model,'tmp_results/' +modelname_to_print+".pt")
    print(modelname_to_print)
    return model
    


def plot_training_loss_validation(train_losses,val_losses,model):
  plt.figure(figsize=(10, 5))
  plt.plot(train_losses, label='Training Loss',color='blue')
  plt.plot(val_losses, label='Validation Loss',color='orange')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Training and Validation Loss')
  plt.legend(['Train', 'Validation'], loc='upper left',bbox_to_anchor=(1.0, 1.00))
  plt.savefig(f'tmp_results/BATCHIERUNNING_training_validation_curve_{model}.pdf',bbox_inches='tight')
  plt.savefig(f'tmp_results/BATCHIERUNNING_training_validation_curve_{model}.png',bbox_inches='tight',dpi=500)
  plt.close()



@dataclass
class NN_DrugComboTheta(Theta):
    def __init__(self,model,X_tensor=None,y_tensor=None):
        self.model = model    
        self.X_tensor = X_tensor
        self.y_tensor = y_tensor
        self.predicted_values = []

    def predict_viability(self, data: ScreenBase) -> np.ndarray:
        if data.treatment_arity != 2:
            raise ValueError("NNDrugCombo only supports data sets with combinations of 2 treatments")
        predicted_viability_values = []

        # Create a TensorDataset
        dataset = TensorDataset(self.X_tensor, self.y_tensor)
        # Create a DataLoader
        batch_size = 64
        train_size = int(0.6 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()  
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                outputs = self.model(inputs).squeeze()  # Squeeze the output tensor to remove any extra dimensions
                predicted_viability_values.extend(outputs.numpy())
        self.predicted_values = np.clip(predicted_viability_values, a_min=0.01, a_max=0.99)
        return np.clip(predicted_viability_values, a_min=0.01, a_max=0.99)

    def predict_conditional_mean(self, data: ScreenBase) -> np.ndarray:
        if len(self.predicted_values) == 0:
            self.predict_viability(data)
        mean = np.mean(self.predicted_values)
        return mean
    
        
    def predict_conditional_variance(self, data: ScreenBase) -> ArrayType:
        if len(self.predicted_values) == 0:
            self.predict_viability(data)
        variance = np.std(self.predicted_values) ** 2
        return variance

    def private_parameters_dict(self) -> dict[str, np.ndarray]:
        params = {"model_state_dict": self.model.state_dict()}
        return params

    @classmethod
    def from_dicts(cls, private_params: dict, shared_params: dict):
        model = linear_nn()
        model.load_state_dict(private_params["model_state_dict"])
        res = cls(model=model)
        return res

class NN_DrugCombo(BayesianModel):
    def __init__(
    self,
    experiment_space: ExperimentSpace,
    n_embedding_dimensions: int,
    predict_interactions: bool = False,
    interaction_log_transform: bool = False,
    intercept: bool = False,
    ):
        self.model = linear_nn()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.X_tensor = None
        self.y_tensor = None
        self.predicted_values = None
        # self.single_effect_lookup = {}
        #I dont think i need these interactions + log_transform + single_effect_lookup
        self.predict_interactions: predict_interactions
        self.interaction_log_transform = interaction_log_transform
        self.n_embedding_dimensions = n_embedding_dimensions

    def get_model_state(self) ->  NN_DrugComboTheta:
        return NN_DrugComboTheta(
            model=self.model,
            X_tensor=self.X_tensor,
            y_tensor=self.y_tensor
        )

    def get_model_parameters(self):
        return self.model.parameters()

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)
        print(f"Model state saved to {file_path}")

    def load_model(self, file_path):
        self.model.load_state_dict(torch.load(file_path))
        self.model.eval()
        print(f"Model state loaded from {file_path}")

    def set_rng(self, rng: np.random.Generator):
        self._rng = rng

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    def _add_observations(self, data: ScreenBase):
        if not (data.observations >= 0.0).all():
            raise ValueError(
                "Observations should be non-negative, please check input data"
            )
        if np.isnan(data.observations).any():
            raise ValueError("NaNs in observations, please check input data")

        sids = data.sample_ids
        dd1s = data.treatment_ids[:,0]
        dd2s = data.treatment_ids[:,1]
        td1s = data.treatment_doses[:,0]
        td2s = data.treatment_doses[:,1]

        obs_tensor = torch.tensor(data.observations, dtype=torch.float32, device=self.device)
        cls_tensor = torch.tensor(sids, dtype=torch.long, device=self.device)
        dd1s_tensor = torch.tensor(dd1s, dtype=torch.long, device=self.device)
        dd2s_tensor = torch.tensor(dd2s, dtype=torch.long, device=self.device)
        td1s_tensor = torch.tensor(td1s, dtype=torch.float32, device=self.device)
        td2s_tensor = torch.tensor(td2s, dtype=torch.float32, device=self.device)

        X_tensor = torch.stack([cls_tensor, dd1s_tensor, dd2s_tensor, td1s_tensor, td2s_tensor], dim=1)

        #standard scaling
        mean = torch.mean(X_tensor)
        std_dev = torch.std(X_tensor)
        self.X_tensor = (X_tensor - mean) / std_dev
        self.y_tensor = obs_tensor
        
    def train(self):        
        model = train_model(self.model,self.X_tensor,self.y_tensor)

    def n_obs(self) -> int:
        return NotImplemented

    def reset_model(self):
        self.model.reset_model()
