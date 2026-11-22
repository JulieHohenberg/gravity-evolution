const modelInfo = {
  'genetic-earth': {name: 'Earth Genetic', detail: 'Genetic algorithm · Earth-trained', short: 'GE', color: '#4d8a6a'},
  'genetic-moon': {name: 'Moon Genetic', detail: 'Genetic algorithm · Moon-trained', short: 'GM', color: '#7888b6'},
  'ppo-earth': {name: 'Earth PPO', detail: 'Reinforcement learning · Earth-trained', short: 'PE', color: '#c78e69'},
  'ppo-moon': {name: 'Moon PPO', detail: 'Reinforcement learning · Moon-trained', short: 'PM', color: '#a27ca5'},
};
const modelsElement = document.querySelector('#models');
const startButton = document.querySelector('#start');
const track = document.querySelector('#track');
const message = document.querySelector('#message');
const statusText = document.querySelector('#race-status');
const timer = document.querySelector('#timer');
const seedInput = document.querySelector('#seed');
const randomSeedButton = document.querySelector('#random-seed');
let planet = 'Earth';
let animation;
let busy = false;

// Keep the button and selected styling in sync with the lineup.
function updateSelection() {
  const selected = [...modelsElement.querySelectorAll('input:checked')];
  modelsElement.querySelectorAll('.model').forEach(label => label.classList.toggle('selected', label.querySelector('input').checked));
  seedInput.disabled = busy;
  randomSeedButton.disabled = busy;
  const seedValid = seedInput.disabled || seedInput.checkValidity();
  startButton.disabled = busy || selected.length === 0 || !seedValid;
  document.querySelector('#setup-note').textContent = !selected.length ? 'Choose at least one racer to begin.' : !seedValid ? 'Enter a whole seed from 0 to 4294967295.' : '';
}

// Load the available trained models from the API.
async function loadModels() {
  try {
    const response = await fetch('/models');
    if (!response.ok) throw new Error('Could not load the racers.');
    const data = await response.json();
    modelsElement.replaceChildren();
    data.models.forEach(model => {
      const info = modelInfo[model.id];
      if (!info) return;
      const label = document.createElement('label');
      label.className = `model${model.available ? '' : ' unavailable'}`;
      const input = document.createElement('input');
      input.type = 'checkbox'; input.value = model.id; input.disabled = !model.available;
      input.checked = model.available && ['genetic-earth', 'ppo-earth'].includes(model.id);
      const icon = document.createElement('span'); icon.className = 'model-icon'; icon.textContent = info.short;
      const copy = document.createElement('span'); copy.className = 'model-name';
      copy.textContent = info.name;
      const detail = document.createElement('small'); detail.textContent = model.available ? info.detail : 'Saved model not found'; copy.append(detail);
      const kind = document.createElement('span'); kind.className = 'model-kind'; kind.textContent = model.id.startsWith('ppo') ? 'PPO' : 'GENETIC';
      label.append(input, icon, copy, kind); modelsElement.append(label);
      input.addEventListener('change', updateSelection);
    });
    updateSelection();
  } catch (error) {
    modelsElement.textContent = 'Could not load racers. Check that the API is running.';
    statusText.textContent = 'Connection problem';
  }
}

// Draw the recorded terrain and robot outlines in world coordinates.
function drawFrame(canvas, racer, frame, world, color) {
  const context = canvas.getContext('2d');
  const width = canvas.clientWidth, height = canvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  }
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const scale = Math.min(27, Math.max(14, height / 7));
  const centerX = frame.body[0].reduce((sum, point) => sum + point[0], 0) / frame.body[0].length;
  const cameraX = Math.max(0, centerX - width * .3 / scale);
  const ground = racer.terrain.reduce((nearest, point) => Math.abs(point[0] - centerX) < Math.abs(nearest[0] - centerX) ? point : nearest, racer.terrain[0]);
  const toX = x => (x - cameraX) * scale;
  const toY = y => height * .76 - (y - ground[1]) * scale;
  context.beginPath(); context.moveTo(toX(racer.terrain[0][0]), height);
  racer.terrain.forEach(point => context.lineTo(toX(point[0]), toY(point[1])));
  context.lineTo(toX(racer.terrain[racer.terrain.length - 1][0]), height); context.closePath();
  context.fillStyle = world === 'Moon' ? '#9ca7a1' : '#98b18b'; context.fill();
  context.strokeStyle = world === 'Moon' ? '#71827d' : '#688d70'; context.lineWidth = 2; context.stroke();
  frame.body.slice().reverse().forEach((polygon, index) => {
    context.beginPath(); polygon.forEach(([x,y], i) => i ? context.lineTo(toX(x), toY(y)) : context.moveTo(toX(x), toY(y)));
    context.closePath(); context.fillStyle = index === frame.body.length - 1 ? color : '#526b5d'; context.fill(); context.strokeStyle = '#30483e'; context.lineWidth = 1.5; context.stroke();
  });
}

// Play all recorded racers against the same simulation clock.
function playRace(data) {
  document.querySelector('#race-world').textContent = `${data.planet.toUpperCase()} GRAVITY`;
  track.className = `track racing${data.planet === 'Moon' ? ' moon-track' : ''}`;
  track.replaceChildren(); track.style.setProperty('--lanes', data.racers.length);
  const lanes = data.racers.map(racer => {
    const lane = document.createElement('div'); lane.className = 'lane';
    const canvas = document.createElement('canvas'); canvas.setAttribute('aria-label', `${modelInfo[racer.model].name} race replay`);
    const label = document.createElement('div'); label.className = 'lane-label'; label.textContent = modelInfo[racer.model].name.toUpperCase();
    const value = document.createElement('div'); value.className = 'lane-result';
    const distance = document.createElement('div'); distance.textContent = '0.00 M';
    const speed = document.createElement('div'); speed.className = 'lane-speed'; speed.textContent = 'Speed: 0.00 m/s';
    value.append(distance, speed);
    const telemetry = document.createElement('div'); telemetry.className = 'lane-telemetry';
    lane.append(canvas, label, value, telemetry); track.append(lane);
    return {racer, lane, canvas, distance, speed, telemetry};
  });
  const finishers = data.racers.filter(racer => racer.status === 'finished');
  const maxStep = finishers.length ? Math.min(...finishers.map(racer => racer.steps)) : Math.max(...data.racers.map(racer => racer.steps));
  const started = performance.now();
  startButton.textContent = 'RACE IN PROGRESS';
  statusText.textContent = 'Race in progress';
  function tick(now) {
    const step = Math.min(maxStep, Math.floor((now - started) / 1000 * data.steps_per_second));
    timer.innerHTML = `${(step / data.steps_per_second).toFixed(1).padStart(4, '0')} <small>SEC</small>`;
    lanes.forEach(({racer, lane, canvas, distance, speed, telemetry}) => {
      const frame = racer.frames.findLast ? racer.frames.findLast(item => item.step <= step) : [...racer.frames].reverse().find(item => item.step <= step);
      drawFrame(canvas, racer, frame || racer.frames[0], data.planet, modelInfo[racer.model].color);
      const stopped = step >= racer.steps;
      const lostAtFinish = finishers.length && step >= maxStep && !data.winners.includes(racer.model);
      lane.classList.toggle('eliminated', (stopped && racer.status !== 'finished') || Boolean(lostAtFinish));
      const ending = stopped ? racer.status.replace('_', ' ').toUpperCase() : lostAtFinish ? 'RACE ENDED' : '';
      distance.textContent = `${(frame || racer.frames[0]).distance.toFixed(2)} M${ending ? ` · ${ending}` : ''}`;
      speed.textContent = `Speed: ${(frame || racer.frames[0]).speed_mps.toFixed(2)} m/s`;
      if (frame && frame.joint_angles) {
        const angles = frame.joint_angles.map(angle => `${Math.round(angle * 180 / Math.PI)}°`).join(' / ');
        const motors = frame.motor_commands ? frame.motor_commands.map(command => command.toFixed(2)).join(' / ') : 'not available';
        telemetry.textContent = `Hip / knee · leg 1, leg 2   |   Angles: ${angles}   |   Motor commands: ${motors}`;
      }
    });
    if (step < maxStep) animation = requestAnimationFrame(tick);
    else {
      const winners = data.winners.map(id => modelInfo[id].name).join(' & ');
      statusText.textContent = `${winners} ${data.winners.length > 1 ? 'tie' : 'wins'} · ${finishers.length ? 'first to finish' : 'furthest distance'}`;
      busy = false; updateSelection(); startButton.firstChild.textContent = 'START RACE ';
    }
  }
  animation = requestAnimationFrame(tick);
}

// Send the selected world and controllers to the race API.
async function startRace() {
  const selected = [...modelsElement.querySelectorAll('input:checked')].map(input => input.value);
  if (!selected.length || busy) return;
  cancelAnimationFrame(animation); busy = true; updateSelection(); message.hidden = true;
  startButton.firstChild.textContent = 'PREPARING RACE '; statusText.textContent = 'Preparing the track';
  try {
    const response = await fetch('/race', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({planet, models:selected, seed:Number(seedInput.value)})});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Could not start the race.');
    playRace(data);
  } catch (error) {
    busy = false; updateSelection(); startButton.firstChild.textContent = 'START RACE ';
    statusText.textContent = 'Race could not start'; message.textContent = error.message; message.hidden = false;
  }
}

document.querySelectorAll('.world').forEach(button => button.addEventListener('click', () => {
  if (busy) return;
  planet = button.dataset.planet;
  document.body.classList.toggle('moon-mode', planet === 'Moon');
  document.querySelectorAll('.world').forEach(item => { const selected = item === button; item.classList.toggle('active', selected); item.setAttribute('aria-pressed', String(selected)); });
  document.querySelector('#race-world').textContent = `${planet.toUpperCase()} GRAVITY`;
  track.className = 'track empty';
  const grid = document.createElement('div'); grid.className = 'track-grid'; grid.setAttribute('aria-hidden', 'true');
  track.replaceChildren(grid);
  statusText.textContent = 'Ready to start';
  timer.innerHTML = '00.0 <small>SEC</small>';
}));
seedInput.addEventListener('input', updateSelection);
randomSeedButton.addEventListener('click', () => {
  let nextSeed;
  do { nextSeed = Math.floor(Math.random() * 4294967296); } while (String(nextSeed) === seedInput.value);
  seedInput.value = String(nextSeed);
  updateSelection();
});
startButton.addEventListener('click', startRace);
loadModels();
