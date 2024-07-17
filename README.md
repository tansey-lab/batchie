# BATCHIE

[![DOI](https://zenodo.org/badge/DOI/10.1101/2023.12.18.572245.svg)](https://doi.org/10.1101/2023.12.18.572245)
![tests](https://github.com/tansey-lab/batchie/actions/workflows/python-unittest.yml/badge.svg)
[![codecov](https://codecov.io/gh/tansey-lab/batchie/graph/badge.svg?token=UPG3BP7U7G)](https://codecov.io/gh/tansey-lab/batchie)

### [batchie.readthedocs.io](https://batchie.readthedocs.io/en/latest/)

Bayesian Active Treatment Combination Hunting via Iterative Experimentation (BATCHIE),
is a framework for orchestrating large-scale combination drug screens through sequential experimental design.

This README is a truncated introduction to the framework. For a more detailed description of the framework,
see the full documentation at https://batchie.readthedocs.io.

## Citation

If you use this code, please cite the [preprint](https://www.biorxiv.org/content/10.1101/2023.12.18.572245v2):

```
A Bayesian active learning platform for scalable combination drug screens
Christopher Tosh, Mauricio Tec, Jessica White, Jeffrey F Quinn, Glorymar Ibanez Sanchez, Paul Calder, Andrew L Kung, Filemon S Dela Cruz, Wesley Tansey
bioRxiv 2023.12.18.572245; doi: https://doi.org/10.1101/2023.12.18.572245
```

Bibtex citation:
```
@article{Tosh2023.12.18.572245,
	author = {Christopher Tosh and Mauricio Tec and Jessica White and Jeffrey F Quinn and Glorymar Ibanez Sanchez and Paul Calder and Andrew L Kung and Filemon S Dela Cruz and Wesley Tansey},
	journal = {bioRxiv},
	title = {A Bayesian active learning platform for scalable combination drug screens},
	year = {2023}}
```

## Installation

### Prerequisites

Nextflow and Python (>=3.11) are required to run the BATCHIE pipeline. The `nextflow` and `python3` commands must be available. 

Instructions for installing nextflow are here (installation time is ~1 minute): 
https://www.nextflow.io/docs/latest/install.html

If you want to use the containerized version of BATCHIE, installing docker is necessary.
Instructions for installing docker are here: https://docs.docker.com/get-docker/

### Option 1: Install Using pip

```
python3 -m venv venv
source venv/bin/activate
pip install git+https://github.com/tansey-lab/batchie
```

On the authors an Apple M1 13" MacBook Pro (2020) with 16GB of RAM, execution of the above command took 23.114 seconds.

### Option 2: Using Docker

The most reproducible way to run BATCHIE is to use our docker container.

We maintain up-to-date and historical docker images on docker hub, you can pull the latest version of BATCHIE with the following command:

```
docker pull jeffquinnmsk/batchie:latest
```

Depending on internet connectivity, this should not take longer than a few minutes. 
The compressed size of the BATCHIE docker image is 5.94 GB at the time of writing.

## Quickstart Demo

Run a quick demo of BATCHIE on a toy dataset.

Install the BATCHIE package in a local Python3 virtual environment:

```
python3 -m venv venv
source venv/bin/activate
pip install git+https://github.com/tansey-lab/batchie
```

Execute the retrospective-mode demo (For explanation of what retrospective-mode means see [here](#retrospective-mode))

```shell
python3 nextflow/scripts/batchie.py \
    --mode retrospective \
    -c nextflow/tests/config/integration_test_retrospective_simulation.config \
    --batch-size 3 \
    --screen nextflow/tests/data/unmasked_screen.h5 \
    --outdir retrospective_sim \
    --n_chains 2 \
    --n_chunks 2 \
    --max_cpus 1 \
    --max_mem 1GB \
    -profile local
```

On the author's an Apple M1 13" MacBook Pro (2020) with 16GB of RAM, 
this command took 45.887 seconds to execute.

Execute the prospective-mode demo (For explanation of what prospective-mode means see [here](#prospective-mode)):

```shell
python nextflow/scripts/batchie.py \
    --mode prospective \
    -c nextflow/tests/config/integration_test_retrospective_simulation.config \
    --batch-size 3 \
    --screen nextflow/tests/data/masked_screen.h5 \
    --outdir prospective_sim \
    --n_chains 2 \
    --n_chunks 2 \
    --max_cpus 1 \
    --max_mem 1GB \
    -profile local
```

On the author's an Apple M1 13" MacBook Pro (2020) with 16GB of RAM, 
this command took 29.687 seconds to execute.

## BATCHIE Data Format

The main data format for BATCHIE is the `batchie.data.Screen` object. You can create a `Screen` object
by organizing your data into numpy arrays and constructing this class.

See below for an example of creating a screen object and saving it to disk:

```python
import numpy as np
from batchie.data import Screen

screen = Screen(
    # The outcome of each experiment,
    # a float array of 0-1 values of shape (n_experiments,)
    # If the experiment has not been observed, this can be set to 0.
    observations=np.array([0.1, 0.2, 0.0, 0.0]),
    # The observation status of each experiment, a boolean array of shape (n_experiments,).
    # If the experiment has been observed, the value is True, otherwise False.
    observation_mask=np.array([True, True, False, False]),
    # The sample name in each experiment, a string array of shape (n_experiments,)
    sample_names=np.array([
        "sample_A",
        "sample_B",
        "sample_C",
        "sample_D"
    ], dtype=str),
    # The plate to which each experiment belongs
    # a string array of shape (n_experiments,)
    plate_names=np.array([
        "plate_A",
        "plate_A",
        "plate_B",
        "plate_B"
    ], dtype=str),
    # The list of drug names used in each experiment 
    # a string array of shape (n_treatments, drug_combination_degree)
    # If the experiment was not a drug combination, 
    # set other drug names to "control" (or whatever the value of `control_treatment_name` is)
    treatment_names=np.array(
        [
            ["drug_A", "drug_B"],
            ["drug_A", "drug_B"],
            ["drug_A", "drug_B"],
            ["drug_A", "control"]
        ], dtype=str
    ),
    # The list of drug doses used in each experiment
    # a float array of shape (n_experiments, drug_combination_degree)
    treatment_doses=np.array(
        [
            [2.0, 2.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [2.0, 0]
        ]
    ),
    # The control treatment name. If this value is observed in the treatment_names array,
    # models will treat it as the absence of any drug.
    control_treatment_name="control"
)

screen.save_h5("my_screen.screen.h5")
```

This will create a file called `my_screen.screen.h5` that contains all the information about the screen, this
is the main input to the BATCHIE pipeline.

## Running BATCHIE

After you have created your BATCHIE `batchie.data.Screen` file,
you can run BATCHIE using our configurable pipeline.

There are two main modes for running BATCHIE, prospective and retrospective.

Our pipline adheres to nf-core standards, and accepts all the
nf-core configuration options, see here: https://nf-co.re/docs/usage/configuration

### Prospective Mode

Assume you have a `batchie.data.Screen` containing one or more
`batchie.data.Plate` s that you have not yet observed.

In prospective mode, BATCHIE will tell you which of those
unobserved plates should be run next. The prospective pipeline has the following steps:

1. Train the specified model given the observed plates
2. Use the trained model to predict on all unobserved plates
3. Use the model predictions on unobserved data to score all the unobserved plates
4. Chose one or more plates with the best scores
5. If ``--batch-size`` is specified, repeat from step 3 until <batch_size> plates have been revealed, if <batch_size> plates have been revealed, repeat from step 1.


From the root of the repository, you would run a command like this:

```
python nextflow/scripts/batchie.py \
    --mode prospective \
    --outdir /tmp/batchie \
    --screen nextflow/tests/data/masked_screen.h5 \
    --batch-size 2 \
    -c nextflow/config/example_prospective.config \
    --max_cpus 1 \
    --max_memory 2G \
    -profile local
```

The output directory will be organized by iteration and plate (if batch_size > 1, then there will be multiple plates per iteration).

In the output directory there will be a simple text file with the id of the best plate to observe next:

```
$ cat /tmp/batchie/iter_0/plate_0/masked_screen/selected_plate
1
$ cat /tmp/batchie/iter_0/plate_1/masked_screen/selected_plate
2
```

### Retrospective Mode

Assume you have a `batchie.data.Screen` where
all `batchie.data.Plate` s are observed, representing data from a high throughput screen
that was run in the past.

In retrospective simulation mode we will run these set-up steps:

1. Mask all of the observations.
2. Sample a fraction of the unobserved experiments and set it aside as "holdout" for evaluating model accuracy.
3. Pick one initial plate to reveal. For the initial plate reveal, one has the option of using a configured heuristic algorithm or picking an existing plate at random.

After these initial steps, we repeat the following series of steps until all plates have been revealed:

1. If all plates have been revealed, exit.
2. Train the specified model given the revealed plate(s)
3. Evaluate the model on the holdout data and save
4. Use the trained model to predict on all unobserved plates
5. Use the model predictions on unobserved data to score all the unobserved plates
6. Chose one or more plates with the best scores to reveal, reveal them
7. If ``--batch-size`` is specified, repeat from step 5 until <batch_size> plates have been revealed, if <batch_size> plates have been revealed, repeat from step 1.


From the root of the repository, you would run a command like this:


```
python3 nextflow/scripts/batchie.py \
    --mode retrospective \
    -c nextflow/config/example_retrospective.config \
    --screen nextflow/tests/data/unmasked_screen.h5 \
    --batch-size 2 \
    --outdir /tmp/batchie \
    --max_cpus 1 \
    --max_memory 2G \
    -profile local
```
The output directory will be similar to prospective mode.


```
$ cat /tmp/batchie/iter_0/plate_0/unmasked_screen/selected_plate
1
$ cat /tmp/batchie/iter_0/plate_1/unmasked_screen/selected_plate
2
```

The pipeline also creates ``model_evaluation.h5`` files for each iteration which save how
the model performed on the holdout set given training on the plates revealed up until that point.
These can be collected for analysis after the simulation completes, for example to create a graph of how
MSE improves as more data is revealed, etc.

## BATCHIE Output Format

The output of the BATCHIE pipeline is a two level hierarchical directory structure.
The top level corresponds to the "rounds" or iterations of the batchie algorithm, and the level below that corresponds
to the "batch" or number of plates revealed in each round.

For example, the output of the demo retrospective pipeline will look like this:

```
retrospective_sim
├── iter_0
│   ├── plate_0
│   │   └── unmasked_screen
│   │       ├── advanced_screen.h5 
│   │       ├── distance_matrix_chunk_0.h5 
│   │       ├── distance_matrix_chunk_1.h5 
│   │       ├── model_evaluation.h5 
│   │       ├── model_evaluation_analysis 
│   │       ├── score_chunk_0.h5 
│   │       ├── score_chunk_1.h5 
│   │       ├── screen_metadata.json 
│   │       ├── selected_plate 
│   │       ├── test.screen.h5 
│   │       ├── thetas_0.h5 
│   │       ├── thetas_1.h5 
│   │       └── training.screen.h5 
│   ├── plate_1
│   │   └── unmasked_screen
│   │       ├── advanced_screen.h5 
│   │       ├── score_chunk_0.h5 
│   │       ├── score_chunk_1.h5 
│   │       ├── screen_metadata.json 
│   │       └── selected_plate 
│   └── plate_2
│       └── unmasked_screen
│           ├── advanced_screen.h5 
│           ├── score_chunk_0.h5 
│           ├── score_chunk_1.h5 
│           ├── screen_metadata.json 
│           └── selected_plate 
└── iter_1
    └── plate_0
        └── unmasked_screen
            ├── advanced_screen.h5 
            ├── distance_matrix_chunk_0.h5 
            ├── distance_matrix_chunk_1.h5 
            ├── model_evaluation.h5 
            ├── model_evaluation_analysis 
            ├── score_chunk_0.h5 
            ├── score_chunk_1.h5 
            ├── screen_metadata.json 
            ├── selected_plate 
            ├── thetas_0.h5 
            └── thetas_1.h5 
```

We ran this demo with a `--batch-size` parameter of 3, this means three plates will be selected in each round (up to the point at which there
are no plates left to reveal). Our demo dataset has 5 plates total, one of which is revealed at the start of the simulation, 
and 4 of which will be chosen by the model. 
As a result the model runs for two rounds, in the first round it chooses 3 plates, and in the second round it chooses the final
plate, and finally it exits as there are no more plates to reveal.

We see there are two directories at the top level of `retrospective_sim`, `iter_0` and `iter_1`, corresponding to each round.
In `iter_0` we see three directories, `plate_0`, `plate_1`, and `plate_2`, corresponding to the three plates chosen in the first round.
In `iter_1` we see one directory, `plate_0`, corresponding to the one plate chosen in the second round.

Before the first plate selection in each round, the underlying model is trained, and then pairwise distance calculations are performed
between the predictions over all the remaining unobserved plates. When the batch size is greater than 1, we do not retrain the model for the
subsequent plates in each round. This is why the number and composition of output files is different for the `plate_0` directory.

### Glossary of Output Files

#### `advanced_screen.h5`

A copy of the screen object with the current plate revealed (only relevant for retrospective mode).

#### `distance_matrix_chunk_<0..n_chunks>.h5`

The pairwise distance matrix between the predictions of the model on the unobserved plates.

Because the pairwise distance matrix can be extremely large, we calculate it in chunks in parallel.

The full distance matrix can be read in with the following code:

```python
import glob
from batchie.distance_calculation import ChunkedDistanceMatrix

distance_matrix = ChunkedDistanceMatrix.concat(
        [ChunkedDistanceMatrix.load(fn) for fn in glob.glob("distance_matrix_chunk_*.h5")]
    )
M = distance_matrix.to_dense()
```

#### `model_evaluation.h5`

Evaluation of the model on the holdout set (only relevant for retrospective mode).

You can inspect this file with the following code:

```python
from batchie.models.main import ModelEvaluation

model_evaluation = ModelEvaluation.load_h5("model_evaluation.h5")
model_evaluation.mse()
```

#### `model_evaluation_analysis`

This folder contains various plots showing the performance of the model on the holdout set (only relevant for retrospective mode).

#### `score_chunk_<0..n_chunks>.h5`

Plate scoring via PDBAL is computationally expensive, so we parallelize the scoring by doing one chunk of plates in each process.
Each of these files contains the scores for one chunk of plates.

The scores for all plates can be read in with the following code:

```python
import glob
from batchie.scoring.main import ChunkedScoresHolder

scores = ChunkedScoresHolder.concat(
        [ChunkedScoresHolder.load_h5(fn) for fn in glob.glob("score_chunk_*.h5")]
    )
scores.get_score(plate_id)
```

#### `thetas_<0..n_chains>.h5`

These objects contained the learned parameters for the model. In each round we train `--n_chains` models, each with a different random seed. 
One `ThetaHolder` object is saved for each trained model.

You can read in all the learned parameters with the following code:

```python
import glob
from batchie.core import ThetaHolder

theta_holder = ThetaHolder(n_thetas=len(glob.glob("thetas_*.h5")))

theta_holders = [theta_holder.load_h5(fn) for fn in glob.glob("thetas_*.h5")]

thetas = theta_holder.concat(theta_holders)
thetas.get_theta(0)
```

The actual Theta object itself that is returned by `get_theta` is implementation specific. Different models have different parameter structures,
so when you implement a new model in the BATCHIE framework you'll need to also implement Theta.

#### `screen_metadata.json`

Metadata about the current state of the screen, for example:

```json
{
    "n_unique_samples": 2,
    "n_unique_treatments": 2,
    "size": 8,
    "n_plates": 5,
    "n_unobserved_plates": 0,
    "n_observed_plates": 5
}
```

#### `selected_plate`

A text file with the name of the plate that was selected in this step:


```shell
$ cat retrospective_sim/iter_1/plate_0/unmasked_screen/selected_plate
3
```

#### `test.screen.h5`

A subset of the Screen which will be held out for model evaluation (only relevant for retrospective mode).

#### `training.screen.h5`

A subset of the Screen which will be used for training the model.


## Computational Cost

The BATCHIE pipeline is generally more CPU/time bound than memory bound. MCMC sampling can take a significant amount
of time to complete. The BATCHIE pipeline does not use GPU or hardware acceleration at this time.

The nf-core standard options ``--max_cpus``, ``--max_memory``, and ``--max_time`` can be used to limit the resources of
individual jobs. ``--max_cpus`` should probably always be set to 1 since no individual jobs utilize multiprocessing
at this time. Parallelizable steps, which include pairwise distance calculation and plate scoring,
are parallelized at the job level. The number of concurrent jobs for these parallelizable steps is controlled by the ``--n_chunks`` parameter.

MCMC sampling is not parallelizable but we allow running multiple MCMC chains to
ensure approximation of the posterior. The number of chains can be controlled with the ``--n_chains`` parameter.

Jobs which fail will be reattempted with higher limits on resources (bounded by the specified maximums). By default 3
retries will be attempted for each job.

## Retrospective Experiments

Below are commands to reproduce our retrospective experiments.

The input ``.screen.h5`` files are available for download here: https://zenodo.org/records/12764821

Warning: These retrospective simulation experiments require a lot of computation time. We recommend these be
run on a high performance computing cluster or cloud computing service. Use the appropriate
`-profile` to specify the computing environment.

### Merck

```shell
python3 nextflow/scripts/batchie.py --mode retrospective \
    -c nextflow/config/merck_retrospective_simulation.config \
    --batch-size 9 \
    --screen merck_2016.screen.h5 \
    --outdir merck_sim \
    --max_memory 64G \
    --max_cpus 1 \
    --max_time 24h
```

### NCI Almanac

```shell
python3 nextflow/scripts/batchie.py --mode retrospective \
    -c nextflow/config/nci_almanac_retrospective_simulation.config \
    --batch-size 18 \
    --screen nci_almanac.screen.h5 \
    --outdir merck_sim \
    --max_memory 64G \
    --max_cpus 1 \
    --max_time 24h
```

### GDSC^2

```shell
python3 nextflow/scripts/batchie.py --mode retrospective \
    -c nextflow/config/gdsc_sq_retrospective_simulation.config \
    --batch-size 36 \
    --screen gdsc_sq.screen.h5 \
    --outdir merck_sim \
    --max_memory 64G \
    --max_cpus 1 \
    --max_time 24h
```