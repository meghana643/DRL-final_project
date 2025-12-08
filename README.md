<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
</head>
<body>

  <h1>Cooperative Multi-Agent RL in Level-Based Foraging (MAPPO)</h1>

  <p>
    This project builds on the codebase from the book:
  </p>

  <blockquote>
    <strong>Stefano V. Albrecht, Filippos Christianos, Lukas Schäfer</strong><br />
    <em>Multi-Agent Reinforcement Learning: Foundations and Modern Approaches</em>, MIT Press, 2024.<br />
    Code: <a href="https://www.marl-book.com" target="_blank" rel="noopener noreferrer">https://www.marl-book.com</a>
  </blockquote>

  <p>
    If you use this codebase in academic work, please cite:
  </p>

  <pre><code>@book{marl-book,
    author = {Stefano V. Albrecht and Filippos Christianos and Lukas Sch\"afer},
    title = {Multi-Agent Reinforcement Learning: Foundations and Modern Approaches},
    publisher = {MIT Press},
    year = {2024},
    url = {https://www.marl-book.com}
}
  </code></pre>

  <p>
    This repository uses the MARL-book codebase (<code>marlbase</code>) to train MAPPO agents on the
    <strong>Level-Based Foraging (LBF)</strong> environment with:
  </p>

  <ul>
    <li>10×10 grid, 3 players, 4 food items (<code>Foraging-10x10-3p-4f-v3</code>)</li>
    <li>Different <strong>field-of-view (FOV / sight)</strong> settings</li>
    <li>Different numbers of <strong>obstacle cells</strong></li>
    <li>Different <strong>player capability levels</strong> (max player level)</li>
  </ul>

  <hr />

  <h2>1. Installation</h2>

  <p>We recommend using a Conda environment:</p>

  <pre><code>conda create -n marlbase python=3.10
conda activate marlbase
  </code></pre>

  <p>Clone and install the marl-book codebase:</p>

  <pre><code>git clone https://github.com/marl-book/codebase.git
cd codebase
pip install -r requirements.txt
pip install -e .
  </code></pre>

  <p>
    Install PyTorch for your system
    (see <a href="https://pytorch.org/get-started/locally/" target="_blank" rel="noopener noreferrer">https://pytorch.org/get-started/locally/</a>).
  </p>

  <p>Install the environments used in this project:</p>

  <pre><code>pip install -U lbforaging rware
  </code></pre>

  <p>
    All algorithms are implemented in <strong>PyTorch</strong> and use the <strong>Gymnasium</strong> API
    (with one reward per agent).
  </p>

  <hr />

  <h2>2. Running MAPPO on Level-Based Foraging</h2>

  <p>All commands below are run from inside the <code>marlbase</code> directory:</p>

  <pre><code>cd marlbase
  </code></pre>

  <h3>2.1 Base MAPPO Training (with video)</h3>

  <p>Example command:</p>

  <pre><code>HYDRA_FULL_ERROR=1 \
python run.py +algorithm=mappo \
    env.name="lbforaging:Foraging-10x10-3p-4f-v3" \
    env.time_limit=25 \
    algorithm.save_interval=10000 \
    algorithm.video_interval=10000 \
    algorithm.video_frames=400 \
    +save_interval=10000 \
    +video_frames=400 \
    +video_interval=10000
  </code></pre>

  <p>This will create an output directory like:</p>

  <pre><code>outputs/lbforaging:Foraging-10x10-3p-4f-v3/mappo/&lt;run_id&gt;/
    config.yaml
    results.csv
    checkpoints/model_sXXXXXX.pt
    ...
  </code></pre>

  <h3>2.2 Training with FOV (sight) Control</h3>

  <p>To change the <strong>field of view</strong> of each agent (e.g., <code>sight=3</code>):</p>

  <pre><code>HYDRA_FULL_ERROR=1 \
python run.py +algorithm=mappo \
    env.name="lbforaging:Foraging-10x10-3p-4f-v3" \
    env.time_limit=25 \
    +env.sight=3 \
    algorithm.save_interval=10000 \
    algorithm.video_interval=10000 \
    algorithm.video_frames=400 \
    +save_interval=10000 \
    +video_frames=400 \
    +video_interval=10000
  </code></pre>

  <p>You can repeat with <code>+env.sight=2</code> and <code>+env.sight=3</code> to compare FOV settings.</p>

  <h3>2.3 Training with Obstacle Cells</h3>

  <p>
    We add obstacles to the LBF grid via the <code>num_obstacles</code> parameter (and a small rendering modification,
    see Section 5).
  </p>

  <p>Example: <strong>10 obstacles, FOV = 3, longer time limit</strong>:</p>

  <pre><code>HYDRA_FULL_ERROR=1 \
python run.py +algorithm=mappo \
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
  </code></pre>

  <h3>2.4 Player Level Configurations (Capability)</h3>

  <p>
    Player capability can be controlled via <code>min_player_level</code> and <code>max_player_level</code> in the environment config.
    For example:
  </p>

  <ul>
    <li>
      All agents level 1:
      <pre><code>... +env.min_player_level=1 +env.max_player_level=1
      </code></pre>
    </li>
    <li>
      Levels in [1, 3]:
      <pre><code>... +env.min_player_level=1 +env.max_player_level=3
      </code></pre>
    </li>
  </ul>

  <p>
    You can create separate runs for different <code>max_player_level</code> values and compare performance and cooperation behavior.
  </p>

  <hr />

  <h2>3. Evaluation</h2>

  <p>To evaluate a trained run using the MARL-book eval script:</p>

  <pre><code>python eval.py path=outputs/lbforaging:Foraging-10x10-3p-4f-v3/mappo/&lt;run_id&gt;
  </code></pre>

  <p>To evaluate a <strong>specific checkpoint</strong> (e.g., at 10,000 steps):</p>

  <pre><code>python eval.py \
    path=outputs/lbforaging:Foraging-10x10-3p-4f-v3/mappo/&lt;run_id&gt; \
    load_step=10000
  </code></pre>

  <p>
    The number of evaluation episodes and other eval settings are controlled by the algorithm’s eval config (Hydra),
    not hard-coded in this script.
  </p>

  <hr />

  <h2>4. Logging and Post-Processing</h2>

  <p>
    By default, the <strong>File System Logger</strong> writes training results to:
  </p>

  <pre><code>outputs/{env_name}/{alg_name}/{run_id}/results.csv
  </code></pre>

  <p>The main metrics used in this project are:</p>

  <ul>
    <li><code>environment_steps</code></li>
    <li><code>mean_episode_returns</code></li>
    <li><code>agent0/mean_episode_returns</code>, <code>agent1/…</code>, <code>agent2/…</code></li>
    <li><code>mean_episode_length</code>, <code>std_episode_returns</code>, etc.</li>
  </ul>

  <p>These are used to create:</p>

  <ul>
    <li>Learning curves (mean return vs. environment steps)</li>
    <li>Cooperation metrics (e.g., std. dev. of agent returns)</li>
  </ul>

  <h3>4.1 Plotting Runs</h3>

  <p>You can use the built-in plotting script:</p>

  <pre><code>python utils/postprocessing/plot_runs.py \
    --source outputs/lbforaging:Foraging-10x10-3p-4f-v3/mappo
  </code></pre>

  <p>
    By default, this plots <code>mean_episode_returns</code> across runs/seeds.
  </p>

  <h3>4.2 Finding Best Hyperparameters / Best Runs</h3>

  <p>To automatically identify the best runs by a metric (default: average total returns):</p>

  <pre><code>python utils/postprocessing/find_best_hyperparams.py \
    --source outputs/lbforaging:Foraging-10x10-3p-4f-v3/mappo
  </code></pre>

  <hr />

  <h2>5. Obstacle Rendering Modifications in <code>lbforaging</code></h2>

  <p>To <strong>visualize obstacles</strong> in Level-Based Foraging, we modify the <code>rendering.py</code> file in the <code>lbforaging</code> package.</p>

  <h3>5.1 File Location</h3>

  <pre><code>~/miniconda3/envs/marlbase/lib/python3.10/site-packages/lbforaging/foraging/rendering.py
  </code></pre>

  <h3>5.2 Add Obstacle Drawing Function</h3>

  <p>Add this method inside the renderer class:</p>

  <pre><code># modification added to draw obstacles
def _draw_obstacles(self, env):
    if not hasattr(env, "obstacles"):
        return  # nothing to draw

    batch = pyglet.graphics.Batch()
    sprites = []  # keep references alive!

    for (row, col) in env.obstacles:
        x = (self.grid_size + 1) * col
        y = self.height - (self.grid_size + 1) * (row + 1)

        if self.img_rock:  # just check directly
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
  </code></pre>

  <h3>5.3 Add <code>rock.png</code> Icon</h3>

  <p>Place a <code>rock.png</code> file in:</p>

  <pre><code>~/miniconda3/envs/marlbase/lib/python3.10/site-packages/lbforaging/foraging/icons
  </code></pre>

  <p>Then in the renderer’s constructor, load the rock image:</p>

  <pre><code>self.img_apple = pyglet.resource.image("apple.png")
self.img_agent = pyglet.resource.image("agent.png")
self.img_rock  = pyglet.resource.image("rock.png")  # modification
  </code></pre>

  <h3>5.4 Call Obstacle Draw Function in <code>render</code></h3>

  <p>Inside <code>render(self, env, return_rgb_array=False):</code>, add:</p>

  <pre><code>def render(self, env, return_rgb_array=False):
    glClearColor(*_WHITE, 0)
    self.window.clear()
    self.window.switch_to()
    self.window.dispatch_events()

    self._draw_grid()
    self._draw_food(env)
    self._draw_players(env)
    self._draw_obstacles(env)  # modification added to draw obstacles
  </code></pre>

  <hr />

  <h2>6. Notes</h2>

  <ul>
    <li>
      This project uses <strong>MAPPO</strong> from the MARL-book codebase with minor environment
      extensions (FOV, obstacles, player-level variations).
    </li>
  
    <li>
      The work is based on and compatible with the main MARL-book codebase:
      <ul>
        <li><a href="https://github.com/marl-book/codebase" target="_blank" rel="noopener noreferrer">https://github.com/marl-book/codebase</a></li>
        <li>Based in part on: <a href="https://github.com/semitable/fast-marl" target="_blank" rel="noopener noreferrer">https://github.com/semitable/fast-marl</a> (by Filippos Christianos)</li>
      </ul>
    </li>
  </ul>

</body>
</html>
