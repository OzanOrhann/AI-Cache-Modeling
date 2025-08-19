# AI-Cache-Modeling  

This project applies **machine learning techniques** to model and predict **CPU cache performance**.  
Synthetic datasets were generated to simulate different cache scenarios.  

## Objective  
- Predict CPU cache hit/miss rates  
- Identify bottlenecks in cache performance  
- Optimize cache efficiency using machine learning  

## Methods  
- **Model:** Single-hidden-layer MLP (feed-forward neural network, ReLU activation)  
- **Loss Function:** Mean Squared Error (**MSELoss**)  
- **Optimizer:** **Adam** (learning rate search: 1e-3, 1e-2)  
- **Hyperparameter Search:** Grid search (lr, hidden_size, batch_size)  
- **Early Stopping:** Patience=5 based on validation error  
- **Evaluation:** MAE and MSE metrics, error visualization  

## Results  
- Error rates were analyzed across different training set sizes.  
- The MLP model successfully captured cache performance patterns under various scenarios.  
