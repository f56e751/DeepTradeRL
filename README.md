# YAI-CON_RL-HFT
The model takes 1-minute OHLCV and Orderbook data as input and learns temporal patterns through a deep learning network combining CNN and LSTM architectures. The reinforcement learning agent makes buy/sell/hold decisions based on processed features, continuously optimizing trading strategies through rewards received from the market environment.

# Core Data Sources:
1-Minute OHLCV Data: Utilizes Open, High, Low, Close, and Volume data at 1-minute intervals for comprehensive price action analysis \\
1-Minute Orderbook Data: Incorporates real-time orderbook depth information to capture market microstructure and liquidity dynamics

# Model Architecture
![image](https://github.com/user-attachments/assets/6383815f-4210-477f-b5de-da2944416933)


# Example
python ppo_agent_tech.py --include_tech True --input_type LSTM

# Result
![image](https://github.com/user-attachments/assets/89a0b507-e069-41ad-8af3-28e2183da3a1)

![image](https://github.com/user-attachments/assets/30a32e34-759f-4ea7-8a45-e52e6ba40e10)
