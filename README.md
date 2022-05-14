# Kuhn Poker Server Client

This repository contains the skeleton code for the PokerBot project agent (server client). The currently implemented agent plays a random game,
and does not yet recognize cards that are dealt as images.

In order to start developing your own agent, you need to create a new python virtual environment (either within your IDE or manually, see this [link](https://code.visualstudio.com/docs/python/environments#_create-a-virtual-environment) for the VSCode as an example):

```bash
# macOS/Linux
# You may need to run sudo apt-get install python3-venv first
python3 -m venv .venv

# Windows
# You can also use py -3 -m venv .venv
python -m venv .venv
```


After you've created you will need to activate the virtual environment every time you start working on your project or create a new terminal session:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

> **_NOTE:_** On the newer versions of Windows running scripts can be disabled by default. You can use `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` command to override this setting and to allow running scripts in the current terminal session.

Then install the required packages:

```bash
# macOS/Linux
pip install -r requirements-linux.txt

# Windows
pip install -r requirements-windows.txt
```

Then generate the game protocol: 

```bash
# macOS/Linux
generate-proto.sh

# Windows
.\generate-proto.sh
```

In order to test your agent implementation and play local games you need to have a local `KuhnPoker` server installed and running in the background. The server backend is available from
[`https://github.com/tue-5ARA0/poker-server-backend`](https://github.com/tue-5ARA0/poker-server-backend). Clone the repository and follow the instructions to start a local server.
In the terminal that runs your local server you will see test player tokens. These tokens represent the agent ids that the server
expects to connect with. You will need to connect an agent to each token. An agent client can be connected to the local server
by opening a new terminal and running:

```bash
python main.py --token "<agent token UUID here>" --play "random"
``` 

Once a second agent is connected (another agent in a second terminal or a bot), the game will start automatically.

> **_NOTE:_** Server waits for a limited amount of time for both players to be connected.

It is possible to run your agent implementation on the public server either against bots or other students. In order to play a public game you need to specify a `--global` flag for the script and to wait for another player:

```bash
python main.py --token "<agent token UUID here>" --global --play "random"
```


In case if you want to play against a specific team you may create a private game with the `--create` argument:

```bash
python main.py --token "<agent token UUID here>" --global --create
```

Server will response with a private coordinator id that you can share and use to connect to a private game, eg.:

```bash
> python main.py --token "<agent token UUID here>" --create
id: "de2c20f1-c6b9-4536-8cb0-c5c5ac816634"

# This command will wait for a limited amount of time until another player is connected
> python main.py --token "<agent token UUID here>" --play "de2c20f1-c6b9-4536-8cb0-c5c5ac816634"
```

You may also rename your team's appearance with the `--rename` option. For full list of available options/arguments use the `--help` flag:

```bash
python main.py --help
```

> **_NOTE:_** You may omit `--token` argument and put your secret token in "token_key.txt" file in the same 
folder as the "main.py" script. 

## Playing a game
[Kuhn poker](https://en.wikipedia.org/wiki/Kuhn_poker) is a simplified version of poker. Upon the start of each round, 
each player is dealt a card as an image. The agent will need to identify its card before deciding on a move. 

> **_NOTE:_**  Our `KuhnPoker` server implementation supports 4-cards variant of the game with the extra `A` card in the game.

First, the server deals cards to each player and calls the `on_image()` method. Next, the server requests an action from each agent in turn by calling the `make_action()` function in `agent.py`. The list of available actions depends
on the current hand and the state of the game. The states are represented in `client\state.py`, where you'll find two class definitions.
The `ClientGameState` tracks the state of the game, from meetup to the win/loss of all chips. The `ClientGameRoundState` tracks
the current round, from deal to showdown. See the assignment introduction presentation for an example. At the end of each game the `on_game_end()` method is always called that acts as a finalizer. In case of multiple games (e.g. during a tournament) a new agent object will be created for each new game.


## Assignment

Your assignment is to equip the poker agent with two main features:

1. A card image classifier, see `on_image()` method;
2. A betting strategy, see `make_action()` method.

For this, you'll need to implement:

- Data set handling (`data_sets.py`);
- An image classifier (`model.py`, `on_image()`);
- An betting strategy (`agent.py`, `make_action()`);
- Tests for your data set handling, classifier, agent and custom functionality (`test\ `);
- Tests for the game and round states (`test\test_client_game_round_state.py`, `test\test_client_game_state.py`).

You can choose your own machine learning toolbox (we suggest TensorFlow with Keras) and are free to modify/create 
files as you see fit. Though it is not forbidden to inspect server communication code, we, however, highly recommend you not to modify `main.py`, `client/events.py`, `client/controller.py` files in order to avoid connection errors. When playing with a global server coordinator between other players a stable interned connection is assumed and reconnection is not allowed (you lose immediatelly if an internet connection lost between you and the server). Some guidance is provided, but you'll need to be creative and implement any additional 
classes/functionality yourself.


### What we look for

The project will be graded on four aspects: clean code, proper tests, team collaboration and effective data handling/AI. 
In the first place we care about clean, correct, well-tested and well-documented code; the performance of your agent is
secondary (although not un-important). More details can be found in the assignment introduction presentation and the
evaluation rubric on Canvas.