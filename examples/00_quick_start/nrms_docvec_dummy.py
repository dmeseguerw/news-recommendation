# TODO make a notebook with it
import torch
# from ebrec.models.newsrec.nrms_docvec import NRMSModel_docvec
from ebrec.models.newsrec_pytorch.nrms import NRMSModel_docvec
from ebrec.models.newsrec.model_config import hparams_nrms
import numpy as np
import torchsummary as summary

DOCVEC_DIM = 300
BATCH_SIZE = 10
HISTORY_SIZE = 20
NPRATIO = 4

#
config = hparams_nrms
config.history_size = HISTORY_SIZE
config.title_size = DOCVEC_DIM

# MODEL:
model = NRMSModel_docvec(hparams=config, newsencoder_units_per_layer=[512, 512])

# Model summary:
summary(model.model, [(HISTORY_SIZE, DOCVEC_DIM), (NPRATIO + 1, DOCVEC_DIM)])

#
his_input_title_shape = (HISTORY_SIZE, DOCVEC_DIM)
pred_input_title_shape = (NPRATIO + 1, DOCVEC_DIM)
label_shape = (NPRATIO + 1,)

# Generate some random input data for input_1
his_input_title = np.array(
    [np.random.rand(*his_input_title_shape) for _ in range(BATCH_SIZE)]
)
# Generate some random input data for input_2
pred_input_title = np.array(
    [np.random.rand(*pred_input_title_shape) for _ in range(BATCH_SIZE)]
)
# Generate some random label data with values between 0 and 1
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
    model.model.train()
    model.optimizer.zero_grad()
    outputs = model.model(his_input_title, pred_input_title)
    loss = model.loss_fn(outputs, label_data)
    loss.backward()
    model.optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

model.model.eval()
# Prediction
with torch.no_grad():
    predictions = model.model(his_input_title, pred_input_title)
    print(predictions)