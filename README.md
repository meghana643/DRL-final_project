Cite the MARL Book:
@book{marl-book,
    author = {Stefano V. Albrecht and Filippos Christianos and Lukas Sch\"afer},
    title = {Multi-Agent Reinforcement Learning: Foundations and Modern Approaches},
    publisher = {MIT Press},
    year = {2024},
    url = {https://www.marl-book.com}
}


This project extends the official Multi-Agent Reinforcement Learning (MARL) book codebase with the following new features:

MAPPO training on Level-Based Foraging (10×10, 3 agents, 4 food)

Limited field-of-view (env.sight)

Obstacle-based environment (env.num_obstacles)

Modified lbforaging rendering to draw obstacles

Evaluation tools and post-processing scripts

All algorithms are implemented in PyTorch and use the Gymnasium API.

Table of Contents

Getting Started

Installation

Running MAPPO

Running MAPPO with Sight

Running MAPPO with Obstacles

Evaluation

Post Processing

Modifications to lbforaging

Contact

Getting Started
Installation

We strongly recommend creating a Conda environment:

conda create -n marlbase python=3.10
conda activate marlbase


Install the project:

pip install -r requirements.txt
pip install -e .


Install PyTorch from official instructions:
https://pytorch.org/get-started/locally/

Install Level-Based Foraging:

pip install -U lbforaging rware

Running MAPPO

Run all commands inside:

cd marlbase

Base MAPPO training:
HYDRA_FULL_ERROR=1 python run.py \
  +algorithm=mappo \
  env.name="lbforaging:Foraging-10x10-3p-4f-v3" \
  env.time_limit=25 \
  algorithm.save_interval=10000 \
  algorithm.video_interval=10000 \
  algorithm.video_frames=400 \
  +save_interval=10000 \
  +video_frames=400 \
  +video_interval=10000

Running MAPPO With Sight
HYDRA_FULL_ERROR=1 python run.py \
  +algorithm=mappo \
  env.name="lbforaging:Foraging-10x10-3p-4f-v3" \
  env.time_limit=25 \
  +env.sight=3 \
  algorithm.save_interval=10000 \
  algorithm.video_interval=10000 \
  algorithm.video_frames=400 \
  +save_interval=10000 \
  +video_frames=400 \
  +video_interval=10000

Running MAPPO With Obstacles

⚠️ Requires modifying rendering.py in lbforaging (see below)

HYDRA_FULL_ERROR=1 python run.py \
  +algorithm=mappo \
  env.name="lbforaging:Foraging-10x10-3p-4f-v3" \
  env.time_limit=75 \
  +env.sight=3 \
  +env.num_obstacles=10 \
  algorithm.save_interval=10000 \
  algorithm.video_interval=10000 \
  algorithm.video_frames=400 \
  +save_interval=10000 \
  +video_frames=400 \
  +video_interval=10000

Evaluation
python eval.py path=outputs/lbforaging:Foraging-10x10-3p-4f-v3/mappo/<run_id>


Evaluate a particular step:

python eval.py \
  path=outputs/lbforaging:Foraging-10x10-3p-4f-v3/mappo/<run_id> \
  load_step=10000

Post Processing
Plot learning curve:
python utils/postprocessing/plot_runs.py \
  --source outputs/lbforaging:Foraging-10x10-3p-4f-v3/mappo

Find best hyperparameters:
python utils/postprocessing/find_best_hyperparams.py \
  --source outputs/lbforaging:Foraging-10x10-3p-4f-v3/mappo

Modifications to lbforaging

We modified the renderer to draw obstacles.

File to edit:
~/miniconda3/envs/marlbase/lib/python3.10/site-packages/lbforaging/foraging/rendering.py

Add this method:
def _draw_obstacles(self, env):
    if not hasattr(env, "obstacles"):
        return

    batch = pyglet.graphics.Batch()
    sprites = []

    for (row, col) in env.obstacles:
        x = (self.grid_size + 1) * col
        y = self.height - (self.grid_size + 1) * (row + 1)

        if self.img_rock:
            sprite = pyglet.sprite.Sprite(self.img_rock, x=x, y=y, batch=batch)
            sprite.update(
                scale_x=self.grid_size / self.img_rock.width,
                scale_y=self.grid_size / self.img_rock.height,
            )
            sprites.append(sprite)
        else:
            square = pyglet.shapes.Rectangle(
                x, y, self.grid_size, self.grid_size,
                color=(200, 0, 0), batch=batch
            )
            sprites.append(square)

    batch.draw()
    self._obstacle_sprites = sprites

Add rock.png to:
~/miniconda3/envs/marlbase/lib/python3.10/site-packages/lbforaging/foraging/icons

Load rock inside constructor:
self.img_rock = pyglet.resource.image("rock.png")

Call obstacle drawing inside render():
self._draw_grid()
self._draw_food(env)
self._draw_players(env)
self._draw_obstacles(env)

Contact

Meghana Chintalapati

Partner Name

Based on: https://github.com/semitable/fast-marl

and www.marl-book.com
