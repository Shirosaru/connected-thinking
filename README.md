# Connected Thinking

This project explores a more connected, idea-bridging form of neural modeling. Instead of treating two inputs as two independent branches that simply collapse into one output, the network builds a shared bridge representation so related concepts can be fused in a more expressive way.

## What is in this repo?

- `simple_nn.py` — a PyTorch-based bridge network with:
  - a two-branch idea bridge
  - a three-branch idea bridge for richer multi-concept fusion
  - demo training runs such as a simple cat/car example
- `concept_graph.py` — a symbolic concept-graph implementation that models idea propagation through weighted, typed relations
- `tests/test_bridge_net.py` — regression tests for the bridge-network behavior
- `concept_graph_export.json` — exported concept graph data
- `network_3d.png` — a sample visualization of the network structure

## Features

- Two-stream bridge fusion
- Three-stream bridge fusion for more complex concept connections
- Small demo datasets for bridging examples like cat/car and multi-concept ideas
- Simple training and visualization pipeline

## Requirements

Python 3.10+ and PyTorch are recommended.

Install dependencies with:

```bash
python3 -m pip install --user torch matplotlib
```

## Run the demos

### Two-branch cat/car demo

```bash
cd /home2/connected_thinking
python3 - <<'PY'
import simple_nn
simple_nn.run_cat_car_demo()
PY
```

### Three-branch idea bridge demo

```bash
cd /home2/connected_thinking
python3 - <<'PY'
import simple_nn
simple_nn.run_triple_bridge_demo()
PY
```

## Run the tests

```bash
cd /home2/connected_thinking
python3 -m pytest -q tests/test_bridge_net.py
```

## Concept graph example

You can also run the symbolic concept graph script:

```bash
cd /home2/connected_thinking
python3 concept_graph.py
```

## Notes

The goal of this project is to move beyond a traditional branch-style propagation setup and explore a more networked, idea-bridging architecture where multiple streams of meaning can converge into a shared representation.
