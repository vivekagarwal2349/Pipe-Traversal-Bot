# **Pipe Traversal Bot**
<b>Title</b>: Implementation of Single and Multi-Agent Deep Reinforcement Learning Algorithms for a Walking Quadruped

## Introduction 

The challenges involved in cleaning and
maintaining Pipes System in various industries
that actually require high human workload, health
risk and much time consuming. Hence there is
much need for such autonomous robot, which can
be deployed in various piplines, to ensure
robustness and travel to the pipe where it is
difficult for a human to reach.

Hence, our work is aimed to train the robot to look
for the defect and sanitising pipelines which saves
money, time and labor effort.

The code is written entirely in Python 3.7.7 and the following Python libraries are required for our code to work.

    pybullet==3.0.6
    numpy==1.18.5
    matplotlib==3.3.2
    tensorflow_probability==0.11.1
    seaborn==0.11.0
    pandas==1.1.4
    tensorflow==2.3.1

Other than this, no additional software is needed for the code to work. The PyBullet Physics Engine is used for simulation using an OpenGL GUI. In this code, we have the following : -

 - A requirement.txt for required python libraries
 - SolidWorks CADs of the SpiderBot
 - SpiderBot URDFs for the SpiderBot
 - Folders for Training Logs & Plots
 - Source Code for the Deep RL Implementation 
 - Training Code to train the SpiderBot with Deep RL
 - Validation Code to test trained models
 - Postprocessing Code to generate plots of training
 

## Folders

### SpiderBot_CADs

This folder contains all the part and assembly files for the SpiderBot. There are options for 3-legged, 4-legged, 6-legged & 8-legged SpiderBots.

### SpiderBot_URDFs

 This folder contains all URDF files and associated STL files for the SpiderBot. There are options for 3-legged, 4-legged, 6-legged & 8-legged SpiderBots.

### Training_Logs & Training_Plots
Folders to store csv file of training data and PDF plots of training.

### Saved_Models
Contains two saved models using DDPG. The FullyTrained Model (1000 episodes) is able to walk well and up to 200 centimetres in the forward direction.

## Source Code

### SpiderBot_Environment.py
This file has the p_gym class. This uses pybullet and loads the plane environment (no obstacles) and the SpiderBot into the physics engine. The code allows an agent to retrieve state observations for a leg or whole SpiderBot and set a target velocity for joints in the SpiderBot. Finally, the code uses information from the physics engine to determine rewards for a time step.

### SpiderBot_Neural_Network.py
This file has the classes for the fully-connected neural networks used. The Tensorflow 2 API is used to develop the neural networks. Depending on the algorithm and number of SpiderBot legs, the neural networks are customised for them. There is all a call method to do a forward propagation through the neural network.

### SpiderBot_Agent.py
This file is a long one, which has all the operations of the agent. It initialises the neural networks based on the algorithm in the constructor. The class also has the functionality to update the target networks for DDPG. Additionally, it has a long list of methods to apply gradients for each one of the algorithms. In these methods, the TensorFlow 2 computational graph and gradient tapes are used to help in backpropagating the loss function. Finally the class also has the functionality to save all models and load all models.

### SpiderBot_Replay_Buffer.py
This file contains the replay_buffer class that handles experience replay storage and operations like logging and sampling with a batch size.

### SpiderBot_Walk.py
This file contains the walk function that is actually responsible for handling all training operations. This is where all the classes interact with each other. The episodes are looped through and the SpiderBot is trained. The training-related data is logged and saved as a csv into the Training_Logs folder while the best models are saved to the Saved_Models folder during training.

### SpiderBot_Postprocessing.py
This file handles the plotting post-processing operations that takes the CSV file from the Training_Logs folder and saves the plot into the Training_Plots folder.

## Main Code

### SpiderBot_Train_Model.py
This file allows the user to set up the training session. In this file, the user can set 3 levels of configuration for training. The general config section has options for choosing algorithms, number of legs, target location, episodes etc. The Hyperparameters config section handles all hyperparameters of the entire training process. The reward structure config provides options for all the scalar rewards. The user must set all of these configs and run the file to train the SpiderBot. TIP: not using a GUI is faster for training, especially if you use a CUDA-enabled NVIDIA GPU.

### SpiderBot_Validation.py
This file allows the user to validate and test a trained model, specially made for the Professors and TAs of SpiderBot to visualise our fully trained model.

## How to train a model?

Unzip the SpiderBot_URDFS.zip file into the same directory. Open up `SpiderBot_Train_Model.py` for editing. The most important parameter is `training_name` that you must define. This is unique to a particular training session and all saved models, logs and plots are based on this `training_name`. After that set up your General Config:

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~ GENERAL CONFIG ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    training_name = "insert_training_name_here"
    model = "DDPG"
    num_of_legs = 8 
    episodes = 1000
    target_location = 1
    use_GUI = True
    do_post_process = True
    save_best_model = True
    save_data = True
    
Following that, set up the configurations for the hyperparameters:

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~ HYPERPARAMETER CONFIG ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    time_step_size = 120./240
    upper_angle = 60
    lower_angle = -60
    lr_actor = 0.00005
    lr_critic = 0.0001
    discount_rate = 0.9
    update_target = None
    tau = 0.005
    max_mem_size = 1000000
    batch_size = 512
    max_action = 10
    min_action = -10
    noise = 1
    epsilon = 1
    epsilon_decay = 0.0001
    epsilon_min = 0.01
Finally, set up the configuration for the reward structure:

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~ REWARD STRUCTURE CONFIG ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    forward_motion_reward = 500
    forward_distance_reward = 250
    sideways_velocity_punishment = 500
    sideways_distance_penalty = 250
    time_step_penalty = 1
    flipped_penalty = 500
    goal_reward = 500
    out_of_range_penalty = 500

Then run the python code

    > python SpiderBot_Train_Model.py
## Team
<table>
    <td align="center">
     <a href="https://github.com/vivekagarwal2349">
    <img src="https://avatars.githubusercontent.com/u/75940729?v=4" width="100px;" alt=""/><br /><sub><b>Vivek Agarwal</b></sub></a><br />
	</td>
	<td align="center">
     <a href="https://github.com/phoenixrider12">
    <img src="https://avatars.githubusercontent.com/u/76533398?s=460&v=4" width="100px;" alt=""/><br /><sub><b>Aryaman Gupta</b></sub></a><br />
	</td>
    <td align="center">
     <a href="https://github.com/Srini-Rohan">
    <img src="https://avatars.githubusercontent.com/u/76437900?v=4" width="100px;" alt=""/><br /><sub><b>Gujulla Leel Srini Rohan</b></sub></a><br />
	</td>
</table>

## Mentor
<table>
  <tr>
    <td align="center"><a href="https://github.com/Raghav-Soni"><img src="https://avatars.githubusercontent.com/u/60649723?v=4" width="100px;" alt=""/><br /><sub><b>Raghav Soni</b></sub></a><br /></a></td>
    </tr>
</table>
