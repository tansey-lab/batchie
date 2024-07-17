# BATCHIE

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

Nextflow and Python (>=3.11) are required to run the BATCHIE pipeline. The `nextflow` and `python3` commands must be available on the `PATH`. 

Instructions for installing nextflow are here (depending on your OS it is as simple as running one command and should not exceed a minute): 
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

On the authors an Apple M1 13" MacBook Pro (2020) with 16GB of RAM, 
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

On the authors an Apple M1 13" MacBook Pro (2020) with 16GB of RAM, 
this command took 29.687 seconds to execute.

## BATCHIE Data Format

The main data format for BATCHIE is the `batchie.data.Screen` object.

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
    # The sample name of each experiment, a string array of shape (n_experiments,)
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
    -profile docker
```

The output directory will be organized by iteration and plate (if batch_size > 1, then theres multiple plates per iteration).

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
    -profile docker
```
The output directory will be similar to prospective mode.


```
$ cat /tmp/batchie/iter_0/plate_0/unmasked_screen/selected_plate
1
$ cat /tmp/batchie/iter_0/plate_1/unmasked_screen/selected_plate
2
```

However there will also be ``model_evaluation.h5`` files for each iteration which save how
the model performed on the holdout set given training on the plates revealed up until that point.
These can be collected for analysis after the simulation completes.

### Computational Cost

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
