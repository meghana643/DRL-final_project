<h1>DRL Final Project:
MAPPO with Field-of-View & Obstacles in Level-Based Foraging</h1>
  
<h2>
  Project Codebase Based on: www.marl-book.com
</h2>

Cite the MARL book using:
```latex
@book{marl-book,
    author = {Stefano V. Albrecht and Filippos Christianos and Lukas Sch\"afer},
    title = {Multi-Agent Reinforcement Learning: Foundations and Modern Approaches},
    publisher = {MIT Press},
    year = {2024},
    url = {https://www.marl-book.com}
}


This project is based on the MARL book codebase and extends it with our own modifications for:

MAPPO training on Level-Based Foraging (10x10, 3 agents, 4 food)

Field-of-view controlled observations (env.sight)

Obstacle-based grid environment (env.num_obstacles)

Custom rendering support for obstacle visualization in lbforaging

Evaluation scripts and post-processing tools

All algorithms are implemented in <i>PyTorch</i> and use the <i>Gymnasium</i> API.

<h1>Table of Contents</h1>

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

We strongly suggest using a virtual environment:

conda create -n marlbase python=3.10
conda activate marlbase


Install dependencies:

pip install -r requirements.txt
pip install -e .


Install PyTorch based on your system using:
https://pytorch.org/get-started/locally/

Install required environments:

pip install -U lbforaging rware

Running MAPPO

All commands are run from inside the marlbase/ directory:

cd marlbase

MAPPO (base version, no sight, no obstacles):
HYDRA_FULL_ERROR=1 python run.py +algorithm=mappo env.name="lbforaging:Foraging-10x10-3p-4f-v3" env.time_limit=25 algorithm.save_interval=10000 algorithm.video_interval=10000 algorithm.video_frames=400 +save_interval=10000 +video_frames=400 +video_interval=10000

Running MAPPO with Sight

We define field-of-view radius (env.sight):

HYDRA_FULL_ERROR=1 python run.py +algorithm=mappo env.name="lbforaging:Foraging-10x10-3p-4f-v3" env.time_limit=25 +env.sight=3 algorithm.save_interval=10000 algorithm.video_interval=10000 algorithm.video_frames=400 +save_interval=10000 +video_frames=400 +video_interval=10000

Running MAPPO with Obstacles

Requires modifications to lbforaging renderer (see below).

HYDRA_FULL_ERROR=1 python run.py +algorithm=mappo env.name="lbforaging:Foraging-10x10-3p-4f-v3" env.time_limit=75 +env.sight=3 +env.num_obstacles=10 algorithm.save_interval=10000 algorithm.video_interval=10000 algorithm.video_frames=400 +save_interval=10000 +video_frames=400 +video_interval=10000

Evaluation

Evaluate model:

python eval.py path=outputs/lbforaging:Foraging-10x10-3p-4f-v3/mappo/<run_id>


Evaluate specific checkpoint:

python eval.py path=outputs/lbforaging:Foraging-10x10-3p-4f-v3/mappo/<run_id> load_step=10000

Post Processing

Plot training curves:

python utils/postprocessing/plot_runs.py --source outputs/lbforaging:Foraging-10x10-3p-4f-v3/mappo


Find best hyperparameters:

python utils/postprocessing/find_best_hyperparams.py --source outputs/lbforaging:Foraging-10x10-3p-4f-v3/mappo

Modifications to lbforaging

The original LBF environment does not visualize obstacles.
We added obstacle support to the renderer.

File to modify
~/miniconda3/envs/marlbase/lib/python3.10/site-packages/lbforaging/foraging/rendering.py

Add this method:
    # modification added to draw obstacles
    def _draw_obstacles(self, env):
        if not hasattr(env, "obstacles"):
            return  # nothing to draw

        batch = pyglet.graphics.Batch()
        sprites = []

        for (row, col) in env.obstacles:
            x = (self.grid_size + 1) * col
            y = self.height - (self.grid_size + 1) * (row + 1)

            if self.img_rock:
                sprite = pyglet.sprite.Sprite(self.img_rock, x=x, y=y, batch=batch)
                sprite.update(
                    scale_x=self.grid_size / self.img_rock.width,
                    scale_y=self.grid_size / self.img_rock.height
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

Add rock.png inside:
~/miniconda3/envs/marlbase/lib/python3.10/site-packages/lbforaging/foraging/icons

Load rock image in constructor:
self.img_rock = pyglet.resource.image("rock.png")

Call obstacle drawing inside render():
self._draw_grid()
self._draw_food(env)
self._draw_players(env)
self._draw_obstacles(env)

Contact

Meghana Chintalapati

Partner Name

Based on: https://www.marl-book.com

and https://github.com/semitable/fast-marl
