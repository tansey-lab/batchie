.. _quickstart:

Quickstart Demo
===============

Run a quick demo of BATCHIE on a toy dataset.

Install the BATCHIE package in a local Python3 virtual environment:

.. code-block:: sh

    python3 -m venv venv
    source venv/bin/activate
    pip install git+https://github.com/tansey-lab/batchie


Execute the retrospective-mode demo (For explanation of what retrospective-mode means see :ref:`running-retrospective`)

.. code-block:: sh

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


On the author's an Apple M1 13" MacBook Pro (2020) with 16GB of RAM,
this command took 45.887 seconds to execute.

Execute the prospective-mode demo (For explanation of what prospective-mode means see :ref:`running-prospective`):

.. code-block:: sh

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


On the author's an Apple M1 13" MacBook Pro (2020) with 16GB of RAM,
this command took 29.687 seconds to execute.
