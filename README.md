# Transformer network for the Dial-a-Ride problem

## Configuration

Clone the repository and setup conda environment.
```bash
git clone https://github.com/staverm/DARPwTransformers.git && cd DARPwTransformers
conda env create --prefix ./env --file environment.yml
conda activate ./env
```

Init clearml
```
clearml-init
```
Then follow instructions

Alternatively copy clearml.conf to your home directory 

Run Nearest Neighbour strategy on the smallest instance of cordeau:
```bash
cd darp
python3 run.py
```

## Usage and parameters

Description of parameters in `run.py`:
- `--epochs`: number of epochs in the experiment 
- `--alias`: is the folder base name for saving (in addition there will be a time stamp)
- `--lr`:  is the adam learning rate
- `--batch_size`:  is the supervision batch size
- `--shuffle`: will mix the datapoints for the training (should be true)
- `--optimizer`: (should be adam)
- `--criterion`: should be crossentropy for my classical supervision
- `--model`: should be Trans18 (not working well with others)
- `--checkpoint_type`: should be best (will actually save the best for train, offline test and best online test with GAP metric)
- `--milestones`: isnt used be you can set it by using the multisteplr scheduler
- `--gamma`: is the reducing facor for the scheduler
- `--dropout`: is dropout rate in transformers
- `--patience`: for the plateau scheduler
- `--scheduler`: is none, plateau or multistep
- `--checkpoint_dir`: diectory to load saved models
- `--total_timesteps`: is a bound to the environment. It just need to be sufficently high, like 10k
- `--monitor_freq` and `--example_freq`: "where used in the monitor callback that i finally gave up even tho its a good way to do train monitoring, just in case" ~ author of orginal code
- `--example_format`: format to save result pictures, should be svg
- `--eval_episodes`: nb of tests each time. In big train it should be reduce as maximum because it's very time consumming
- `--verbose`: "(probably very wrongly used in the code...)" ~ author of orginal code
- `--max_step`: maximum of steps used by envirement which will be replecated by model. Should be large number
- `--nb_target`: the parameter of the problem instance. In problem definition a2-16 16 stands for number of targets
- `--image_size`: is the size of the square in witch the instance takes place. Should be 10 to match cordeau's sizes
- `--nb_drivers`: the parameter of the problem instance. In problem definition a2-16 2 stands for number of drivers
- `--env`: should be DarSeqEnv (but actually the flag is not used in the code, it always right) ~author of orginal code
- `--dataset`: "can be '' if you have all the right parameters, it will create and then use the one dataset that matches the specs" ~ author of orginal code
- `--rootdir`: where the code is
- `--reward_function`: defined for rl or for the testing take name from `utils.py`
- `--clearml`: bool if use clearml experiment saving
- `--data_size`: is the nb of points in the dataset. If you change it, it will create a new dataset that matches this size, from scratch. This size is in the data specs as well.
- `--typ`: is a short for some combination of parameters, my best results are with typ=33 (others are only in orginal code)
- `--timeless`: "will cut out the time constraints of the problem but I havent used it a lot, im not sure it is still working" ~ author of orginal code
- `--embed_size`: the size of the embeddings in the transformer (the output of the decoder beeing (1+2*t+d) * embed size
- `--rl`: "is the time step where rl is taking over the train process, shouldn't  be used." ~ author of orginal code
- `--vacab_size`: "is changed hardcoded in the code, but it is just the max input sequence of the transformer i think"  ~ author of orginal code
- `--supervision_function`: is nn or rf (rf not used in this repository) according to the data you use as supervision
- `--balanced_dataset`:  dataset will try out different types of balancing coded in supervised_trainer - only used with rf so not used in this code
- `--num_layers`: is the classical parameters for transformer to know how menny time we repeat the blocks
- `--forward_expansion`: parameter of transformer 
- `--heads`: number of transformers heads
- `--pretrain`: orginal authors parameter to try different approach. Shouldn't be used 
- `--datadir`: is for where your data is.
- `--augmentation`: "is defined in utils.objects.supercisiondataset
I think it is a good way to enhance the data, probably there are other ways to do this or ways to do it better. The idea it to apply eucludian transformation to the datapoints and there solutions so the time and space distances dont change (and the supervision stays correct) but the data points value do change"  ~ author of orginal code

## Code structure 

- `run.py`: main file to set up the experiments. Run it with arguments provided above
- in ```train``` folder there are functions used to train model the most important is `supervised_trainer.py` it has:
  - `run` function which is main training loop 
  - `updating_data` which creates training set and test set with help of `generate_supervision_data`
  - `train` function which for every point in dataset collects information about environment, chooses next action and updates the weights according to the opimizer
- other files as `evaluations.py` and `save_examples` are helper functions to the trainer
- modularized transformer model is in ```models``` folder
- in ```strategies``` there are defined strategies to learn especially `nn_strategy` (nearest neighbours)
- in ```envirements``` different envirements are defined, we use `DarSeqEnv` which is sequential representation of problem
