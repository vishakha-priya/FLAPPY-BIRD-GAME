# FLAPPY-BIRD-GAME
In this project , I trained an agent to play Flappy Bird game using Reinforcement Learning specially the Deep-Q-Network.

The agent interact with the environment by taking actions(like flap wings or do nothing ) and receives rewards based on its performance,such as surviving longer or passing obstracles

In this to solve the two problems of DQN i.e correlated samples and moving target we have used Experience Replay (state ,action,reward,next_state,termination), policy network and target network.

Experience Replay is implemented through python Deque for optimizations mini batches from these is used to avoid correlated sample which also improve the training .

policy network is used for training and updation while target network is a seperate network used to compute stable target Q values.

The weight of the Target network is periodically updated by copying the weights from policy network
