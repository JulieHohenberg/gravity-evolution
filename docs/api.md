# Race API

The API uses [FastAPI](https://github.com/fastapi/fastapi). [app.py](../app.py) accepts JSON requests and returns JSON results. It runs races without opening a desktop window.

Start it from the project folder:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to try the endpoints. `GET /models` lists the four controllers and shows which saved files are available. Genetic controllers come from `controller-earth.npy` and `controller-moon.npy`; PPO controllers use the most recently saved `final-model.zip` for each planet in `runs/`.

`POST /race` accepts a planet and one to four different models. For example, this races an Earth-trained genetic controller against a Moon-trained PPO controller on Earth:

```json
{
  "planet": "Earth",
  "models": ["genetic-earth", "ppo-moon"]
}
```

The other model IDs are `genetic-moon` and `ppo-earth`. The training planet in the model name can be different from the race planet. Each controller runs on the same seeded course for up to 1,600 steps. The response includes distance, total reward, finish status, and the winner by greatest forward distance. It also includes terrain and robot body positions at regular steps so a website can replay the race. Races are simulated before the response is sent.

The trained model files are kept locally and ignored by Git. To run the API on another computer or host, the selected model files must also be available there.
