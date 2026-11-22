# gravity-evolution

This first step shows the same walking robot in Earth and Moon gravity using [CARL](https://github.com/automl/CARL) and its Box2D physics environment.

Earth gravity is 9.81 m/s² and Moon gravity is 1.62 m/s²

## Set up on a Mac

Open Terminal in this project folder. Python 3.9 or newer is required. Run these commands one at a time:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install swig
python -m pip install -r requirements.txt
```

A virtual environment keeps this project's packages separate from other projects. SWIG is a tool needed to build Box2D. If the installation asks for Apple's command line developer tools, install them with `xcode-select --install` and retry the last command.

## Run the simulation

Each time you open a new Terminal, activate the virtual environment and start the program:

```bash
source .venv/bin/activate
python simulation.py
```

In the window, press **1** for Earth, **2** for the Moon, **R** to reset, **Space** to pause or resume, and **Esc** to quit. Switching environments starts a fresh run.

## Train a controller

Train on Earth and watch the result:

```bash
python train_genetic.py
python simulation.py --controller controller-earth.npy
```

Use `python train_genetic.py --planet Moon` to train under Moon gravity. You can load either controller in the viewer and switch between planets.

To train with PPO instead, run `python train_rl.py --planet Earth` or `python train_rl.py --planet Moon`. The command prints the folder containing the saved model. Watch it with `python simulation.py --ppo runs/<run-folder>/final-model.zip`.

See [the neural network](docs/neural-net-controller.md), [genetic algorithm](docs/genetic-algo.md), and [reinforcement learning](docs/rl-algo.md) for details.

## Race API

Run `python -m uvicorn app:app` and open `http://127.0.0.1:8000` to use the race interface. Open `http://127.0.0.1:8000/docs` to test the API. See [api.md](docs/api.md) for the model choices and request format.

## UI collaborator

The UI was polished in collaboration OpenAI's GPT-6.1 Terra and the [oil-ui skill](https://github.com/oil-oil/oil-ui/blob/main/SKILL.md).

## Deploy on Vercel

Connect the repository to Vercel and deploy it as a FastAPI app. Vercel uses Python 3.12 by default. The website and race API are served by `app.py`.

- [requirements.txt](requirements.txt) lists the Python packages. Linux uses CPU-only PyTorch because races don't need GPU libraries.
- [uv.toml](uv.toml) makes SWIG available when Vercel builds Box2D and points the installer to the CPU PyTorch packages.
- [vercel.json](vercel.json) allows a race request up to 60 seconds.
- [models/](models/) contains copies of the four trained controllers so a deployment can run races. The app uses local training results when present and these copies otherwise. Replace the copies to deploy newly trained models; their contents are included in the repository.

Vercel's [Large Functions beta](https://vercel.com/changelog/vercel-functions-can-now-be-up-to-5-gb-in-package-size) may be needed if the installed packages exceed the standard size limit.

## Python files

- [simulation.py](simulation.py) sets up Earth and Moon gravity and displays the robot.
- [controller.py](controller.py) turns the robot's sensor readings into motor commands using a neural network (puts neural net into INFERENCE MODE).
- [train_genetic.py](train_genetic.py) trains the controller with a genetic algorithm.
- [train_rl.py](train_rl.py) trains a controller with PPO.
- [app.py](app.py) provides the race API.
- [ui/](ui/) contains the race interface.
- [tests/test_controller.py](tests/test_controller.py) checks the motor commands and fitness scoring.
- [tests/test_rl.py](tests/test_rl.py) checks PPO training and its saved files.
- [tests/test_api.py](tests/test_api.py) checks race requests and results.
