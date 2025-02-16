# RNN-Name-Classification
This project implements a simple Recurrent Neural Network (RNN) for classifying names into categories based on the characters in the name. The model is trained to predict the origin or category of a given name (e.g., predicting the country of origin for names like "Albert" or "Satoshi"). It uses character-level inputs, making it suitable for tasks where the relationship between characters (such as letter sequences) is significant. The model is trained using PyTorch, with a focus on applying RNNs for sequence classification.

## Key Features
Character-level Classification: The RNN is trained on names represented as sequences of individual characters.
Category Prediction: For a given name, the model predicts the most likely category or origin (e.g., country or ethnicity).
Training Loop: The model is trained using stochastic gradient descent and the negative log likelihood loss (NLLLoss).
Interactive Prediction: Once trained, users can input a name to predict its category using the trained RNN model.
Model Saving: After training, the model parameters are saved to a file for later use.

## File Structure
├── rnn_model_250000.pth         # Trained model checkpoint

├── utils.py                     # Utility functions (e.g., data loading, tensor conversion)

├── rnn_model.py                 # Implementation of the RNN and training code

├── streamlit_app.py             # Streamlit app implmentation

├── README.md                    # GitHub project documentation

└── .gitignore                   # Git ignore file

# How It Works
### RNN Model: The core of the project is the RNN class, which is a simple Recurrent Neural Network built using PyTorch's nn.Module. It consists of:

### Input Layer: Takes character-level input encoded as tensors.
### Hidden Layer: A recurrent hidden layer that processes sequences.
### Output Layer: Outputs the category prediction. The forward() method defines how the input data passes through the network, with a softmax layer applied to the output for probability calculation.
Data Processing:

Character to Tensor Conversion: Each character of the input name is converted to a tensor using the letter_to_tensor() function.

- Sequence Conversion: The line_to_tensor() function converts an entire name into a sequence of tensors, which the RNN processes one character at a time.
- Training: The training loop uses the train() function to perform forward passes, calculate the loss, and backpropagate the error for model optimization. The model is trained using Stochastic Gradient Descent (SGD) and the negative log-likelihood loss function (NLLLoss).

- Prediction: After training, the model is used to predict the category of a name. The predict() function takes an input name, processes it, and returns the predicted category.

- Loss Plotting: During training, the model's loss is plotted every 1000 iterations, which helps in tracking the convergence of the model.

- Saving the Model: After training, the model is saved using torch.save() for future use, allowing easy reloading and prediction on new names.

# Demonstration:

![image](https://github.com/user-attachments/assets/7fb5acb0-9a75-48b4-991b-67d7018ff27a)
![image](https://github.com/user-attachments/assets/32996254-05d5-4dac-b902-d17630fce111)

