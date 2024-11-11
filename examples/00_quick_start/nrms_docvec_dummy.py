# TODO make a notebook with it
from ebrec.models.newsrec.nrms_docvec import NRMSModel_docvec
from ebrec.models.newsrec.model_config import hparams_nrms
import numpy as np
import torch
from torchsummary import summary

DOCVEC_DIM = 300
BATCH_SIZE = 10
HISTORY_SIZE = 20
NPRATIO = 4

# Configuration
config = hparams_nrms
config.history_size = HISTORY_SIZE
config.title_size = DOCVEC_DIM

# MODEL:
model = NRMSModel_docvec(hparams=config, newsencoder_units_per_layer=[512, 512])

# Create dummy inputs
his_input_title_shape = (BATCH_SIZE, HISTORY_SIZE, DOCVEC_DIM)
pred_input_title_shape = (BATCH_SIZE, NPRATIO + 1, DOCVEC_DIM)

his_input_title = torch.tensor(np.random.rand(*his_input_title_shape), dtype=torch.float32)
pred_input_title = torch.tensor(np.random.rand(*pred_input_title_shape), dtype=torch.float32)

# Print the summary
summary(model.model, [(HISTORY_SIZE, DOCVEC_DIM), (NPRATIO + 1, DOCVEC_DIM)])

# Generate some random label data with values between 0 and 1
label_shape = (NPRATIO + 1,)
label_data = np.zeros((BATCH_SIZE, *label_shape), dtype=np.float32)
for row in label_data:
    row[np.random.choice(label_shape[0])] = 1

# Convert NumPy arrays to PyTorch tensors
label_data = torch.tensor(label_data)

# Print the shapes of the input data to verify they match the model's input layers
print(his_input_title.shape)
print(pred_input_title.shape)
print(label_data.shape)

# Training loop
for epoch in range(10):
    model.optimizer.zero_grad()
    outputs = model.model(his_input_title, pred_input_title)
    loss = model.loss_fn(outputs, label_data)
    loss.backward()
    model.optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Prediction
with torch.no_grad():
    predictions = model.model(his_input_title, pred_input_title)
    print(predictions)