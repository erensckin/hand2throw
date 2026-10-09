# Hand2Throw

This repo explores the use of fine-tuned SmolVLA for a dynamic throwing task & a built latent action model (LAM) to explore data retrieval from the user videos. Built upon LeRobot, LIBERO and SmolVLA. 

## The Task

<p align="center">
  <img src="media/throw_side.gif" width="300" alt="Throw at 1.00 m">
</p>

<p align="center"><em>Side camera: the policy throws at 1.00 m, from its cameras alone.</em></p>

An extension of the LIBERO-Object task 4: "pick up the ketchup and place it in the basket". The task explored is putting the ketchup in the basket, but the basket's distance from the robot varies from 0.7m to 1.0m away. The robot needs to switch between placing & throwing based on whether it can reach the basket, and adjust its throw strength based on distance. The clutter in the scene jitters by ±1cm. 

## VLA Fine-Tuning

The SmolVLA used in the original LIBERO tasks (lerobot/smolvla_libero) is further fine-tuned with user collected data on the new task. The fine-tuning only trained the action expert, the vision-language backbone staying frozen. 

### Data Collection

To fine-tune the new task, user data was collected using a mix of teleoperation through hand-tracking and pre-determined motions. For the grip and movement, teleoperation is conducted. Using 2 cameras, the user's hand is tracked in 3D. The distance between the index finger and thumb determines the Franka Panda's grip (binary). Upon a successful grip, the user presses t on their keyboard, which conducts the pre-determined throw motion based on basket placement. This was done as camera low fps & motion blur make tracking at high speeds difficult, meaning webcam teleop is insufficient for highly dynamic actions. The throw demonstrations use privileged strength based on true position of the basket. The policy does not receive this privilege and needs to infer strength and distance from cameras. The throw action exceeds the maximum speed of the joints, and thus this specific throw motion is from simulation testing only. 

An initial 205 demos were recorded, of which 5 are excluded due to bad training data(such as knocking basket over during placement, throwing instead of placing). 

The dataset is available at https://huggingface.co/datasets/erensckin/hand2throw-teleop-demos

<p align="center">
  <img src="media/demo-teleop.gif" width="600" alt="Data-Collection Demo">
</p>

<p align="center"><em>Data collection demonstration for teleoperation and pre-determined trajectory throwing.</em></p>

### Fine-Tune Explorations & Evaluation

The fine-tuning methods and their respective success rates over 210 episodes (70 episodes per seed, 3 seeds) is given. The evaluation tests the baskets at 0.70m, 0.75m, 0.80m. 0.85m, 0.90m, 0.95m and 1.00m, 10 per distance per seed. No training was done for 0.75m, 0.85m or 0.95m therefore these test the continuous strength inference by the VLA model. The evaluation was conducted at 50 action predictions per chunk. 


| Method | Description | Grip Success Rate | Throw Success Rate | Success Rate |
|---|---|---|---|---|
| Main model | 200 teleop demos, 20k steps | 83 % | 49 % | 44 % |
| Fewer demos | 100 demos, 20k steps | 83 % | 45 % | 41 % |
| Teleop-matched demos | 56 teleop demos (the scenes kept for the video-inferred set), 10k steps | 88 % | 58 % | 49 % |
| Video-inferred demos | the same 56 scenes, hand actions inferred from hand video, 10k steps  | 33 % | 23 % | 20 % |
| Extra training (control) | main + 5k steps on the same data | 80 % | 42 % | 41 % |
| Self-improvement | main + 5k steps on demos + its own successful rollouts | 65 % | 35 % | 31 % |
| Self-improvement + throw up-weighting | main + 5k steps on demos + its own successful rollouts + throw up-weighting | 40 % | 17 % | 16 % |
| No fine-tuning | pretrained lerobot/smolvla_libero (35 episodes only) | 0 % | 0 % | 0 % |


Models available at https://huggingface.co/erensckin/hand2throw-smolvla

### Execution Time Adaptation Evaluation

SmolVLA's chunks were adapted by executing a given amount of actions before replanning.


| Method | Description | Grip Success Rate | Throw Success Rate | Success Rate |
|---|---|---|---|---|
| Shorter chunks (25) | main model, 25 actions per chunk | 89 % | 49 % | 50 % |
| Shorter chunks (10) | main model, 10 actions per chunk| 78 % | 46 % | 47 % |
| Re-plan before the throw | main model, chunks of 25, fresh plan right before the throw | 87 % | 52 % | 51 % |
| Short chunks during the throw | main model, chunks of 25, then 10 once the throw starts | 88 % | 50 % | 50 % |
| Async inference, 2-step latency | main model, next chunk computed while the arm moves (100 ms) | 79 % | 34 % | 31 % |
| Async inference, 3-step latency | main model, next chunk computed while the arm moves (150 ms) | 75 % | 35 % | 30 % |
| Random layouts | main model, objects at random positions (70 episodes) | 1 % | 0 % | 0 % |


### Evaluation Anaylsis

The policy successfully places at 0.70m and throws beyond, and its throw strength matches the demos, including the unseen distances. It infers the basket distance from vision. Model hesitates only at the 0.75m boundary (3% success)

- **Main Model**: Showed decent success rates, but struggled in the throws mainly due to increased scatter of lands compared to privileged thrower used in demos.
- **100 Demonstrations**: Within noise of main, shows that more demonstrations are not neccessary. 
- **Matched Demos (teleop & video-inferred)**: shows that video-inferred has considerably less success, which may be due to the mapping used for training or the LAM's ability to infer motion. More on this in the LAM section. The teleop matched data obtained better results than main. Possibly within noise, but may imply that the LAM & successful video inference may have acted as a filter to select the cleaner demonstration data, obtaining better results upon training on those given demos. 
- **Extra training**: Within noise of main, shows that beyond 20k steps the success rate stays around constant. 
- **Self-improvement**: Attempted to increase success by following successful VLA-made demos, hypothesizing it would further fit the learning. Gave worse results, possibly due to worse quality demos with a part of success being based on luck, as well as the possibility of overfitting and reducing adaptability for generating actions from unfamiliar states. 
- **Self-improvement + Throw up-weighting**: Worse results by far. Throw up-weighting may have broken the continuity between grip and throw by showing throw only videos, deeming the throw very fragile to starting position and deeming it better to follow one continuous motion. 
- **Shorter Chunks**: Reducing chunk size makes no measurable difference to grasping or throwing. 
- **Asynchronous Inferece**: Done to emulate real-robot performance, where the continuous throw would not be able to re-infer mid throw. Grip success decreases slightly as latency increases, and throw success drops more relative to its starting percentage. Plans not joining up and the consequent small jump in direction/speed is expected to reduce success whilst affecting the throw relatively more. Additionally, policy trained on synchronous inference; motivates re-training of asynchronous inference. Could

Increasing policy performance can be attempted in multiple ways:
- Improve starting data: Obtain throw data with RL policy instead of pre-determined. May reduce training quality, but may also reduce scatter of throws compared to pre-determined, which had its own lower std scatter compared to VLA. 
- Improve throw after trainig: Use residual RL to correct the VLA's actions during the throw to reduce execution scatter. 
- Asynchronous Inference Improvement: Training policy on asychronous inference and attempting real-time chunking is hypothesizes to bring the async success closer to that of the sync (main) success.

## Latent Action Model


## Run it Yourself

### Install the project

```bash
git clone https://github.com/erensckin/Hand2Throw && cd Hand2Throw
uv sync --locked
uv run python -c "import libero.libero"   # first import asks a question: answer N
```

Set up the cameras: a webcam facing you, and optionally a phone on your left side, looking across your hand. For the phone, install the DroidCam app, connect it to the same Wi-Fi as the computer, and note the IP address it shows.
Practise without recording:
bash
   uv run python scripts/teleop.py --no-record --camera2 http://PHONE_IP:4747/video
Record:
bash
   uv run python scripts/teleop.py --camera2 http://PHONE_IP:4747/video
In the window:
press SPACE so the robot follows your hand;
pinch your thumb and index finger to grasp the ketchup;
for a close basket, carry the ketchup over it and release;
for a far basket, press t to throw.
Successful episodes save automatically. Press d to discard an episode and q to quit.
Train on what you recorded: bash scripts/train_throw.sh full (it reads data/throw_ketchup by default).

Without a phone, leave out --camera2 .... The webcam alone then controls forward/back and up/down.



EXTRAS: 

The SmolVLA model is fine-tuned on user-recorded teleoperation data where the simulation robot is controlled by tracking the user's hand through 2 cameras to obtain 3-D tracking. The recorded simulation data is then used to fine-tune a "put the ketchup in the basket" task, where the basket's distance from the robot varies from the robot being able to place or needing to throw the ketchup into the basket. The LAM is explored to attempt to retrieve robot actions from hand recordings without direct teleoperation data, create simulation recordings from the real-life videos, and train the SmolVLA on the new LAM-generated videos. “Train the SmolVLA on the new LAM-generated videos” isn’t accurate. The LAM infers actions from your hand video, those actions are replayed in simulation, and the policy trains on those demos. Also, “base recordings” is unclear; “videos of my hand” says it plainly.