Running BATCHIE
===============

After you have created your batchie :py:class:`batchie.data.Screen` file (see here: :ref:`data-format`),
you can run batchie using our configurable nextflow pipeline.

There are two main modes for running batchie, :ref:`running-prospective` and :ref:`running-retrospective`.

Our pipline adheres to nf-core standards, and accepts all the
nf-core configuration options, see here: https://nf-co.re/docs/usage/configuration

Prerequisites
-------------

You must have Nextflow and docker installed to run the examples shown here.
See :ref:`install-nextflow-docker` for more details.


.. _running-prospective:

Prospective
-----------

Assume you have a :py:class:`batchie.data.Screen` containing one or more
:py:class:`batchie.data.Plate` s that you have not yet observed.

In prospective mode, batchie will tell you which of those
unobserved plates should be run next.

From the root of the repository, you would run a command like this:

.. code-block:: bash

    nextflow run main.nf \
        --mode prospective \
        --outdir /tmp/batchie \
        --screen nextflow/tests/data/masked_screen.h5 \
        -c nextflow/config/example_prospective.config \
        --max_cpus 1 \
        --max_memory 2G \
        -profile docker


The output will be a json file indicating the plate_ids to run next:

.. code-block:: bash

    $ jq '.' /tmp/batchie/prospective/selected_plates.json
    {
      "selected_plates": [
        1
      ]
    }

.. _running-retrospective:

Retrospective
-------------

Assume you have a :py:class:`batchie.data.Screen` where
all :py:class:`batchie.data.Plate` s are observed, representing data from a high throughput screen
that was run in the past.

In retrospective simulation mode we will run these set up steps:

#. Mask all of the observations.
#. Sample a fraction of the unobserved experiments and set it aside as "holdout" for evaluating model accuracy.
#. Pick one initial plate to reveal. For the initial plate reveal, one has the option of using a configured heuristic algorithm or picking an existing plate at random.

After these initial steps, we repeat the following series of steps until all plates have been revealed:

#. Train the specified model given the revealed plate
#. Evaluate mean squared error of the model on the holdout data
#. Use the trained model to predict on all unobserved plates
#. Use the model predictions on unobserved data to score all the unobserved plates
#. Chose one or more plates with the best scores to reveal, reveal them and repeat from step 1.

Because this pipeline is recursive, we are not able to run it completely in nextflow. We have a small
python wrapper around nextflow in ``nextflow/scripts/run_retrospective_simulation.py`` for running this
pipeline. All that is required to run this script is a recent version of python3 and to have nextflow installed.
The batchie python package does not need to be installed locally.

From the root of the repository, you would run a command like this:

.. code-block:: bash

    python3 nextflow/scripts/run_retrospective_simulation.py \
        -c ../batchie/nextflow/config/example_retrospective.config \
        --screen nextflow/tests/data/unmasked_screen.h5 \
        --outdir /tmp/batchie \
        --max_cpus 1 \
        --max_memory 2G \
        -profile docker

The output directory will contain a directory for each step of the simulation.

In each step directory there will be a json file indicating which plates were
selected at each step up to that point, and the loss (mean squared-error of prediction
vs. holdout data) at each step:

.. code-block:: bash

    $ jq '.' /tmp/batchie/2/unmasked_screen/simulation_tracker_output.json
    {
      "plate_ids_selected": [
        [
          0
        ],
        [
          1
        ]
      ],
      "losses": [
        0.04701452633941148,
        0.020094239180636828
      ],
      "seed": 12
    }
