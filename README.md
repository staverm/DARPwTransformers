# Transformer network for the Dial-a-Ride problem
The Dial-a-Ride Problem (DARP) is a complex combinatorial problem considered in the operational research field. Given a set of drivers and a stream of tasks, where each task is a ride between two points under some time constraints, one has to find a mapping that assigns a driver to the given task. This problem is highly relevant in today's economy as delivery services have gained large popularity and become widespread. It is usually solved by using hand-crafted heuristics inspired by solutions for the Travelling Salesman Problem. This repository contains the code for a transformer network capable of cloning a supervision policy on DARP instances. This provides evidence that such an architecture is capable of understanding the intricacies of DARP and thus similar architectures can likely be applied to other NP-hard problems in operational research.

## Configuration

Clone the repository and setup conda environment.
```bash
git clone https://github.com/staverm/DARPwTransformers.git && cd DARPwTransformers
conda env create --prefix ./env --file environment.yml
conda activate ./env
```

Initialize clearml if you wish to use it.
```
clearml-init
```
Then follow instructions, alternatively copy clearml.conf to your home directory.

Run the script which starts:
- supervision dataset generation using Nearest Neighbor strategy
- supervised training
- model evaluation
```bash
cd darp
./start.sh
```

## Usage and parameters

Description of parameters of `run.py` and `start.sh`:
- `--epochs`: number of training epochs 
- `--alias`: folder name used for saving experiment metadata
- `--lr`:  adam learning rate
- `--batch_size`:  supervision batch size
- `--shuffle`: shuffles datapoints used for training (should be true)
- `--optimizer`: (should be adam)
- `--criterion`: should be crossentropy for classical supervision
- `--model`: alias of the model to be used, should be Trans18 (not working well with others)
- `--checkpoint_type`: should be best, saves the best model weights for train, offline_test and online+test
- `--milestones`: deprecated, but can be set by using the multisteplr scheduler
- `--gamma`: reducing factor for the scheduler
- `--dropout`: dropout rate in transformers
- `--patience`: plateau scheduler
- `--scheduler`: none, plateau or multistep
- `--checkpoint_dir`: directory from where to load saved models
- `--total_timesteps`: bound to the environment, needs to be sufficently high ~ 10k
- `--monitor_freq` and `--example_freq`: deprecated
- `--example_format`: format to save result pictures, should be svg
- `--eval_episodes`: nb of tests performed on each step of evaluation, on big trains should be reduced as it is very time consumming
- `--verbose`: deprecated
- `--max_step`: maximum number of steps used by environment which will be replicated by the model, should be a large number
- `--nb_target`: number of targets, parameter of the problem instance
- `--image_size`: size of the the square in which the instance takes place, should be 10 to match cordeau's sizes
- `--nb_drivers`: number of drivers, parameter of the problem instance
- `--env`: should be DarSeqEnv
- `--dataset`: should be '', path to cached supervised dataset
- `--rootdir`: project root directory
- `--reward_function`: defined for rl or other testing function from the `utils` package
- `--clearml`: if present, clearml is used to save experiment metadata
- `--data_size`: number of points in the dataset. Changing it will create a new dataset that matches this size, from scratch.
- `--typ`: used for debug and testing: a short for combination of parameters, best results are with typ=33 (others are present in orginal code)
- `--timeless`: deprecated, cuts time constraints of the problem
- `--embed_size`: the size of the embeddings in the transformer, the output of the encoder beeing (1+2*t+d) * embed size
- `--rl`: deprecated, the time step where rl is taking over the train process
- `--vacab_size`: max input sequence of the transformer, currently hardcoded
- `--supervision_function`: the supervision policy trying to be learned. Should be nn for nearest neighbour or rf for restricted fragment (rf not used in this repository)
- `--balanced_dataset`: balancing mode used for creating test and validation sets
- `--num_layers`: number of layers for transformer
- `--forward_expansion`: parameter of transformer 
- `--heads`: number of transformers heads
- `--pretrain`: deprecated
- `--datadir`: path to data directory 
- `--augmentation`: turns on data augmentation as described in original paper

## Code structure 

- `run.py`: main file to run experiments. Run it with arguments provided above
- ```train``` folder contains functions used to train models, most notably `supervised_trainer.py` which has:
  - `run` function which is the main training loop 
  - `updating_data` which creates training set and test set with help of `generate_supervision_data`
  - `train` function which for every point in dataset collects information about environment, chooses next action and updates the weights according to the opimizer
  - `evaluations.py` and `save_examples` define helper functions for the trainer
- modularized transformer model is in ```models``` folder
- ```strategies``` folder contains supervision strategies trying to be learned, most notably `nn_strategy` (nearest neighbours)
- ```environments``` folder contains different environment implementations, we use `DarSeqEnv` which is a sequential representation of given darp instance
