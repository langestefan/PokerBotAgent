# Data handling
## Dataset specification

| Input data specification |                   |
|----------------|-----------------------------|
| Labels         | `J`, `Q`, `K`, `A`          |
| Image size     |32 x 32 pixels                 |
| Rotation angle |0 to 15 degrees              |
| Noise level    |0 to 1 with 0.1 stepsize     |

## Implementation
### Feature extraction
The extracted feature shape is (32*32) which meets the input requirement of the model. The feature values are divided by 255 to normalize the values in the range [0,1].
### Data set generation
The generated data set contains an amount of images, each image has one out of four difference letters from `J`, `Q`, `K`, `A`. Each image is ramdomly rotated by 0 to 15 degrees with noise level 0 to 1. The name of image is organized as `label_i.png`, where the 'lable' demonstrates the letter showned in the image and the 'i' represents the generated order of the image.
### Data set load and separation
 - Load the generated data set.
 - Extract features and labels from each image in the data set.
 - Separate the data set into training and validation sets with extracted features and labels.

## DVC
DVC is a tool that helps to version control generated data sets and trained models. It is also very convenient to share the data sets and trained model with team members. Furthermore DVC was chosen because it's easy to setup, has a familiar git-like interface and is free and open-source with great community support. 

For our implementation we have opted to use DVC in combination with SSH remote storage. The docker container uses a lightweight version of ubuntu with only a basic filesystem and the SSH daemon (`sshd`) running. 

The DVC SSH remote storage is set up using a docker container with the following docker file:

```dockerfile
FROM ubuntu:latest
RUN apt update && apt install  openssh-server sudo -y
RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 <user> 
RUN  echo '<user>:<password>' | chpasswd
RUN service ssh start
EXPOSE <port>
CMD ["/usr/sbin/sshd","-D"]
```

The docker container is hosted on the NAS of a teammember and uses simple user/password authentication with the option to change to key-pair based authentication. Once the docker container has started a simple SSH command can be used to connect to the remote storage:
```bash
ssh -p <port> <user>@<remote>
```

Since the volume of the docker container is mounted on the NAS, docker containers can be created and destroyed without the need to reinstall anything. The dockerfile takes care of the installation of the required packages and the SSH daemon.

## Data versioning
Model and dataset versions can be tracked with DVC. To track the data version, the data is first stored in on the local machine. Then the following commands are used to store the data on the remote storage:

```bash
dvc add dataset  # track dataset version in DVC
git add dataset.dvc # track DVC dataset file in git
git commit -m "Add dataset version 1"
dvc push
```
Now the train/test datasets can be pulled by any team member using the following DVC command:

```bash
dvc pull --remote ssh-storage data_sets\
```

## Generated data set overview
The following table lists the generated data sets.

| Generated dataset |    Nr. of images   |
|----------------|-----------------------------|
| Training set                | 10000 (80/20 train/val split)          |
| Test set                    | 2000                                   |
| Unit test test set          | 4                                      |
| Unit test training set      | 4                                      |


# Model
## How to build, train and evaluate a model?
A model is built by using the `build_model()` in the `model.py` file. This function simply returns an instance of the `PasaNet()` class defined in `pasanet_nn.py`. When we have a model, we can train it using the `train_model()` function. We have to give the model we want to train, indicate the amount of validation data and note if we want to save the trained model into a file. After training the model can be evaluated using the `evaluate_model()` function, this return the accuracy on the test set, which contains 2000 images. To simplify training and testing a seperate file `train.py` is created. In this file the model is build, trained and tested. To run it simply use:

```bash
python train.py
```

This will build a model trained with 2000 validation images and the trained model won't be saved, however this function has optional arguments to change if the model should be saved and the amount of validation images:

```bash
usage: train.py [-h] [-v N_VAL] [-s SAVE_MODEL]

Train and evaluate model

optional arguments:
  -h, --help     show this help message and exit
  -v N_VAL       number of validation samples
  -s SAVE_MODEL  decide if model should be saved or not
```

## Our model structure
The model used to classify the card images is a simple Convolutional Neural Network (CNN) called PasaNet. It consist of 3 convolutional layers followed by a ReLU activation layer, a batch normalization layer and a pooling layer. A fully connected layer followed by a ReLU activation layer completes the model. The model is built using the Pytorch framework. The model only takes a few milliseconds to identify an image, well within the latency requirement. Below, a summary of the model is given.

```bash
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 34, 34]              80
              ReLU-2            [-1, 8, 34, 34]               0
       BatchNorm2d-3            [-1, 8, 34, 34]              16
         MaxPool2d-4            [-1, 8, 17, 17]               0
            Conv2d-5           [-1, 32, 17, 17]           2,336
              ReLU-6           [-1, 32, 17, 17]               0
       BatchNorm2d-7           [-1, 32, 17, 17]              64
         MaxPool2d-8             [-1, 32, 8, 8]               0
            Conv2d-9             [-1, 16, 8, 8]           4,624
             ReLU-10             [-1, 16, 8, 8]               0
      BatchNorm2d-11             [-1, 16, 8, 8]              32
        MaxPool2d-12             [-1, 16, 4, 4]               0
          Flatten-13                  [-1, 256]               0
           Linear-14                    [-1, 4]           1,028
             ReLU-15                    [-1, 4]               0
================================================================
Total params: 8,180
Trainable params: 8,180
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.48
Params size (MB): 0.03
Estimated Total Size (MB): 0.52
----------------------------------------------------------------
```

## Model versioning
Since the first model already showed promising results (accuracy 85%+), our first model is also our final model. Only the hyperparameters are optimized. Model versions are just like the data not directly tracked in Git but in DVC. Adding the model to DVC is a similar process to adding the data. The following commands are used to store the model on the remote storage:

```bash
dvc add PasaNet  # track model called 'PasaNet' in DVC
git add Pasanet.dvc # track DVC model file in git
git commit -m "Add model version 1"
dvc push
```

We do have other versions of the model, these are not directly tracked in Git, but in DVC together with the data. For example in commit [70bc11c](https://github.com/tue-5ARA0-2021-Q3/pokerbot-pokerbot-group-8/tree/70bc11c47ef5c1613d8ee378f9f1005e00561147) there was a change to PasaNet V2 but because it did not perform better than the original PasaNet the commit was reverted, but it is still accessible through Git and DVC. In the section below this will be shown how.

## Use the pre-trained model
Since we don't want the user to necessarily train their own model, there will be a pre-trained model supplied. The model is trained for 10 epochs in batches of 50 samples using the Adam optimizer and cross-entropy loss on 8000 training images and 2000 validation images, all images have a noise level randomly assigned between 0.0 and 1.0 to make it robust against the unknown noise level of the server. This model is not stored directly in Git(Hub) since it is tracked in DVC together with the data. To access it simply use:

```bash
dvc pull --remote ssh-storage PasaNet
```
This pulls the model called PasaNet from a remote storage called ssh-storage. If we want to pick a different version of the model, for example the previously mentioned PasaNet V2 in commit [70bc11c](https://github.com/tue-5ARA0-2021-Q3/pokerbot-pokerbot-group-8/tree/70bc11c47ef5c1613d8ee378f9f1005e00561147). Then we first have to re-integrate this specific commit to our branch, we do this through the "cherry-pick" command. After this our `PasaNet.dvc` file should have changed and we can pull the model version from DVC.
```bash
git cherry-pick 70bc11c #re-integrate commit where the model version changed
dvc pull --remote ssh-storage PasaNet #pull model version from DVC
```


## Hyperparameter optimization
For hyperparameter optimization a developer tool called *Weights & Biases* is used. Hyperparameter optimization is still done by hand but using this tool the training and test accuracy of the model are visualized in graphs. This gives us indications how to change the hyperparameters. For example to determine the amount of epochs to train on the figures below are used. We can clearly see when we train for 100 epochs that after a certain amount of iteration the training loss keeps decreasing but that the validation loss is increasing again, and the validation accuracy is also not increasing. We should stop at the amount of iteration where the validation loss starts increasing. Therefore, we concluded that training for 10 epochs is the right choice. This can also be seen on the results of the test set. If we train for 10 epochs instead of 100 the test loss is much lower.

![trainvalloss](https://github.com/tue-5ARA0-2021-Q3/pokerbot-pokerbot-group-8/blob/classifier/figures/trainloss.png)

![trainaccuracy](https://github.com/tue-5ARA0-2021-Q3/pokerbot-pokerbot-group-8/blob/classifier/figures/accuracy.png)

![testaccuracy](https://github.com/tue-5ARA0-2021-Q3/pokerbot-pokerbot-group-8/blob/classifier/figures/test_loss_final.png)

## Final model
Using *Weights & Biases* several combinations of learning rates, batch sizes, amount of epochs and optimizers are evaluated. The model achieves a 94.5% accuracy on the test set, the model is only not able to identify extremey noise images like the image shown below. 

![noisy](https://github.com/tue-5ARA0-2021-Q3/pokerbot-pokerbot-group-8/blob/classifier/figures/bad_image.png)

In the end the following data and hyperparamets are used for the final model:

### Hyperparametes
- Number of epochs: 10
- Learning rate: 0.0001
- Optimizer: Adam
- Batch size: 50
- Loss: Cross-Entropy

### Results
- Accuracy on test set: 94.5%

# Strategy
## Kuhn Poker
As an example we can take the game tree of Kuhn poker containg the cards ``[J, Q, K]`` and the actions ``[Fold, Call, Bet, Check]`` which is visualized below. In the game tree all the card dealings, possible actions and resulting rewards are added to the tree as nodes. The game with four cards is similar to this with the increment of additional branches for new possible card dealings. 

![Kuhn game tree](https://github.com/tue-5ARA0-2021-Q3/pokerbot-pokerbot-group-8/blob/agent_strategy/strategy/figure/kuhn_game_tree.PNG)


## Algorithm: Vanilla Counterfactual Regret Minimization
Based on the game tree, the idea of Vanilla Counterfactual Regret Minimization (CFR) can be described in the following steps:
- Select the action we want to take
- Calculate the reward received
- Calculate counterfactual rewards, i.e. other possible received rewards
- Substract counterfactual rewards from actual rewards to get regret, e.g. if the left bottom "fold" in the tree is selected, then the regret value of the "call" behind it should be -2 - 1 = -1
- Store regrets in a table
- Sum up regrets and normalize to get a strategy
- Select the action with the highest utility (i.e. the highest normalized regret) as the next action for each information set in the game 

## Implementation
### Result map
The result map is defined for constructing the game tree in constants.py, including:
- All possible card dealings for two players, i.e. all combinations of two cards in ``[J, Q, K, A] ``
- The results of all card dealings at showdown
- The error cases received from the server (this is only for error checking)

### Constructing the game tree
The construction of the game tree is implemented in kuhn_state_tree.py. By calling `KuhnRootChanceGameState()` with the input of all card dealings, the game tree corresponding to the result map can be built. The related classes can be summarized as below:
- `GameStateBase()`: The basic info of a node containing its parent node, current player and availiable actions
- `KuhnPlayerMoveGameState()`: The move info contained in the current node. This includes `actions_history`, `cards`, `children`, `public_card` and `_information_set` in addition to the static info. 

    ```bash
    Attributes
    ----------
    actions_history : str
        Previously made actions of both players. Actions in the list alternate between players, i.e., the first element is the first action of player A, and the second element is the first action of player B, etc. The last element of actions_history is the last action made by your opponent. If the child node is the first to move, actions_history will be empty.
    cards : str
        The cards dealing in the current round, e.g. "KJ" means that player A has "K", and Player B has "J".
    children: KuhnPlayerMoveGameState
        -to_move: It turns to player A or B. Taking the turn by -to_move. 
        actions_history + [a]: add the current action in the actions_history. Each possible action would be added to the tree as a branch.
        cards: keep the info of the cards.
        __get_actions_in_next_round: get the available choices for the next action.
    public_card:str 
        Current card in hand for the player of this node, e.g. if cards = KJ and to_move = A, which means that the current node records the move info of player A, then public_card = K. If to_move = B, then public_card = J.
    _information_set: str
        The information set of a player containing the card in hand and the actions history.
        E.g. if to_move = A, public_card = K, and actions_history = [BET, CALL], then _information_set = [K.BET.CALL]. In this example, the current node describes that the player A has K on hand, and player A BET at first, then player B CALL before this node. 
    ```
- `KuhnRootChanceGameState()`: constructing the tree by calling `KuhnPlayerMoveGameState()`

### Computation of CFR utility
Given the game tree, we can compute the CFR utility of each node using the following functions in crf_algorithm.py:
- `VanillaCFR()`: Creates a CFR object based on the game tree
- `VanillaCFR().run`: Recursively updates the utility of each node for 1000 iterations 
- `compute_nash_equilibrium()`: Normalizes the obtained utility to get the final strategy which approximates the Nash Equilibrium

More in-depth explanations for some terms used above:

- Counterfactual utility is the weighted sum of utilities for all subgames (each rooted in a single game state) contained in the current information set. The weights are normalized counterfactual probabilities of reaching these states. 
- Nash Equilibrium is a strategy profile (a set of strategies shared by all players) in which no single player has an incentive to deviate. It represents a point of equilibrium between players, where no player benefits from changing their strategy. A game can be considered to have a Nash-Equilibrium strategy profile if changing one player's strategy provides no additional value (in terms of utility) when the other player plays their original strategy (they do not change it). In other words, both players always play the best corresponding response action at any given point in the game. If two no-regret algorithms play a zero-sum game against eachother for T iterations with average regret less than a constant value ϵ, then their average strategies approximate the Nash Equilibrium. Here, we approximate it by recursively computing it for 1000 iterations.

### Obtaining the strategy dictionary
By calling `create_strategy()` the output strategy is created and saved in the file agent_strategy.json which is stored in the root directory of the project. This function is only called if either the agent strategy file doesn't exist or when the game tree is updated.

## Results
The obtained strategy dictionary shows the relationship between the current inf_set and the utility of the next available actions. Part of such a dictionary is shown below. The state of the game defines our location in the tree, at any given point we always opt to choose that action which has the highest utility.

```json
".":{
  "KQ":0.08333333333333333,
  "KJ":0.08333333333333333,
  "QK":0.08333333333333333,
  "QJ":0.08333333333333333,
  "JK":0.08333333333333333,
  "JQ":0.08333333333333333,
  "JA":0.08333333333333333,
  "QA":0.08333333333333333,
  "KA":0.08333333333333333,
  "AJ":0.08333333333333333,
  "AQ":0.08333333333333333,
  "AK":0.08333333333333333
},
".K.":{
   "BET":0.005863978746673386,
   "CHECK":0.9941360212533267
},
".Q.BET":{
   "FOLD":0.6838352457521532,
   "CALL":0.3161647542478469
},
".Q.CHECK":{
   "BET":0.003928571428571429,
   "CHECK":0.9960714285714286
},
".K.CHECK.BET":{
   "CALL":0.9997485253580443,
   "FOLD":0.0002514746419557558
},
...
``` 

## Evaluation & Improvements

### Performance in the Kuhn poker game
In general, the strategy approximates Nash Equilibrium, so it is supposed to perform well in the poker game. Through testing in local and online games, this strategy indeed improves the win rate and its stability compared with the initial random method. There is a rough comparison between the CFR strategy and the random choice when running the game 10 times.

| 4-card | CFR | Random |
| :----: | :----: | :----: |
| WIN | 8 | 4 |
| DEFEAT  | 2 | 6 | 

### The number of cards
The implemented strategy supports both 3-card and 4-card games as the dictionary is generated based on ["J","Q","K","A"], i.e. we can always find any of these four cards in the resulting dictionary no matter how many cards are dealt in the actual game. However, it will perform better in 4-card cases than 3-card, because the utility is calculated based on the relationship between four cards, resulting in an inaccurate estimation for 3-card games sometimes. The comparison between 3-card games and 4-card games in 10 random attempts is shown in the below table.

| Cards   | ["J","Q","K","A"] | ["J","Q","K"] |
| :----: | :----: | :----: |
| WIN | 8 | 6 |
| DEFEAT  | 2 | 4 | 

### The number of players
If the rules of actions in a round are not changed, the strategy dictionary is robust for running the poker game against any number of players, because the information set of any player only includes the card in hand and the action history in a round, which is always a valid key in the dictionary, but the utility will be inappropriate except for the game with two players. The reason is that the generated Kuhn game tree only considers all the comparisons between two cards and the alternation between two players in a round and that the utility of each action is calculated based on the tree. 

### Possible improvements in flexibility
The obtained utility is only appropriate for certain cases because the current generation of the strategy dictionary is based on constants.py, where the cards and players are set manually according to our prior knowledge of the game but not auto-generated according to the game request. It could be improved by passing the related arguments (i.e. `cards` and `players`) to `PokerAgent()`. Additionally, it will be safer that the content of `cards` is a list of specific cards used in the game rather than only the number of cards. In this way, the Kuhn game tree and the corresponding utility dictionary can be generated more adaptively and flexibly.

We did not do these as there would be many modifications of arguments in controller.py, agent.py, and main.py meanwhile the server does not support this kind of game request. Briefly, the implementation of the game is that a bot sends a game request with the below arguments to the server and then makes an action after receiving the response. We can see that there is no argument about the number of players and specific cards. Even though we changed our bot code, it will not work if the server does not match the modification. 

```bash
    parser.add_argument('--token', help = 'Player\'s token', default = None)
    parser.add_argument('--play', help = 'Existing game id, \'random\' for random match with real player, or \'bot\' to play against bot', default = 'random')
    parser.add_argument('--cards', help = 'Number of cards used in a game', choices=[ '3', '4' ], default = '3', type = str)
    parser.add_argument('--create', action = 'store_true', help = 'Create a new game', default = False)
    parser.add_argument('--local', dest = 'server_local', action = 'store_true', help = 'Connect to a local server', default = False)
    parser.add_argument('--global', dest = 'server_global', action = 'store_true', help = 'Connect to a default global server', default = False)
    parser.add_argument('--server', help = 'Connect to a particular server')
    parser.add_argument('--rename', help = 'Rename player', type = str)
```

## Usage of the strategy in `make_action()`
- If agent_strategy.json does not exist, create it by calling `create_strategy()`;
- Load agent_strategy.json;
- Look for the corresponding action with the maximum utility according to the current inf_set in the dictionary, i.e., the next action. 

## References
[Counterfactual Regret Minimization code](https://github.com/int8/counterfactual-regret-minimization)


[CFR algorithm](https://int8.io/counterfactual-regret-minimization-for-poker-ai/)


[Kuhn tree](https://justinsermeno.com/posts/cfr/)




