# Transformer network for the Dial-a-Ride problem

## Usage

Clone the repository and setup conda environment.
```bash
git clone https://github.com/staverm/DARPwTransformers.git && cd DARPwTransformers
conda env create --prefix ./env --file environment.yml
conda activate ./env
```

Run Nearest Neighbour strategy on the smallest instance of cordeau:
```bash
cd darp
python3 run.py
```
