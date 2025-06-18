# Alpha Zero General (any game, any framework!)

## Installation

- Python version 3.10.18
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Alpha Zero General

A simplified, highly flexible, commented and (hopefully) easy to understand implementation of self-play based reinforcement learning based on the AlphaGo Zero paper (Silver et al). It is designed to be easy to adopt for any two-player turn-based adversarial game and any deep learning framework of your choice. A sample implementation has been provided for the game of Othello in PyTorch and Keras. An accompanying tutorial can be found [here](https://suragnair.github.io/posts/alphazero.html). A concise description of our algorithm can be found [here](https://github.com/suragnair/alpha-zero-general/raw/master/pretrained_models/writeup.pdf). We also have implementations for many other games like GoBang and TicTacToe.

To use a game of your choice, subclass the classes in ```Game.py``` and ```NeuralNet.py``` and implement their functions. Example implementations for Othello can be found in ```othello/OthelloGame.py``` and ```othello/{pytorch,keras}/NNet.py```. 

```Coach.py``` contains the core training loop and ```MCTS.py``` performs the Monte Carlo Tree Search. The parameters for the self-play can be specified in ```main.py```. Additional neural network parameters are in ```othello/{pytorch,keras}/NNet.py``` (cuda flag, batch size, epochs, learning rate etc.). 

### Experiments
We trained a PyTorch model for 6x6 Othello (~80 iterations, 100 episodes per iteration and 25 MCTS simulations per turn). This took about 3 days on an NVIDIA Tesla K80. The pretrained model (PyTorch) can be found in ```pretrained_models/othello/pytorch/```. You can play a game against it using ```pit.py```. Below is the performance of the model against a random and a greedy baseline with the number of iterations.
![alt tag](https://github.com/suragnair/alpha-zero-general/raw/master/pretrained_models/6x6.png)

### Citation

If you found this work useful, feel free to cite it as

```
@misc{thakoor2016learning,
  title={Learning to play othello without human knowledge},
  author={Thakoor, Shantanu and Nair, Surag and Jhunjhunwala, Megha},
  year={2016},
  publisher={Stanford University, Final Project Report}
}
```

Some extensions have been implented [here](https://github.com/kevaday/alphazero-general).
