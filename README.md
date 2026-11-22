# gravity-evolution

Machine Learning and AI (MESA8414) final project. This first step shows the same walking robot in Earth and Moon gravity using [CARL](https://github.com/automl/CARL) and its Box2D physics environment.

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
