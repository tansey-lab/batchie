Running BATCHIE
===============

After you have created your BATCHIE :py:class:`batchie.data.Screen` file (see here: :ref:`data-format`),
you can run BATCHIE using our configurable pipeline.

There are two main modes for running BATCHIE, :ref:`running-prospective` and :ref:`running-retrospective`.

Our pipline adheres to nf-core standards, and accepts all the
nf-core configuration options, see here: https://nf-co.re/docs/usage/configuration

Prerequisites
-------------

Because our pipeline is recursive, we are not able to run it completely in nextflow. We have a small
python wrapper around nextflow in ``nextflow/scripts/batchie.py`` for running this
pipeline. All that is required to run this script is a recent version of python3 and to have nextflow installed.
The batchie python package does not need to be installed locally, since nextflow can launch all jobs in our
container if the ``-profile docker``, ``-profile singularity``, ``-profile podman``, etc. flags are used.


.. _running-prospective:

Prospective
-----------

Assume you have a :py:class:`batchie.data.Screen` containing one or more
:py:class:`batchie.data.Plate` s that you have not yet observed.

In prospective mode, BATCHIE will tell you which of those
unobserved plates should be run next. The prospective pipeline has the following steps:

#. Train the specified model given the observed plates
#. Use the trained model to predict on all unobserved plates
#. Use the model predictions on unobserved data to score all the unobserved plates
#. Chose one or more plates with the best scores
#. If ``--batch-size`` is specified, repeat from step 3 until <batch_size> plates have been revealed, if <batch_size> plates have been revealed, repeat from step 1.


From the root of the repository, you would run a command like this:

.. code-block:: bash

    python nextflow/scripts/batchie.py \
        --mode prospective \
        --outdir /tmp/batchie \
        --screen nextflow/tests/data/masked_screen.h5 \
        --batch-size 2 \
        -c nextflow/config/example_prospective.config \
        --max_cpus 1 \
        --max_memory 2G \
        -profile docker


The output directory will be organized by iteration and plate (if batch_size > 1, then theres multiple plates per iteration).

In the output directory there will be a simple text file with the id of the best plate to observe next:

.. code-block:: bash

    $ cat /tmp/batchie/iter_0/plate_0/masked_screen/selected_plate
    1
    $ cat /tmp/batchie/iter_0/plate_1/masked_screen/selected_plate
    2

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

#. If all plates have been revealed, exit.
#. Train the specified model given the revealed plate(s)
#. Evaluate the model on the holdout data and save
#. Use the trained model to predict on all unobserved plates
#. Use the model predictions on unobserved data to score all the unobserved plates
#. Chose one or more plates with the best scores to reveal, reveal them
#. If ``--batch-size`` is specified, repeat from step 5 until <batch_size> plates have been revealed, if <batch_size> plates have been revealed, repeat from step 1.


From the root of the repository, you would run a command like this:

.. code-block:: bash

    python3 nextflow/scripts/batchie.py \
        --mode retrospective \
        -c nextflow/config/example_retrospective.config \
        --screen nextflow/tests/data/unmasked_screen.h5 \
        --batch-size 2 \
        --outdir /tmp/batchie \
        --max_cpus 1 \
        --max_memory 2G \
        -profile docker

The output directory will be similar to prospective mode.

.. code-block:: bash

    $ cat /tmp/batchie/iter_0/plate_0/unmasked_screen/selected_plate
    1
    $ cat /tmp/batchie/iter_0/plate_1/unmasked_screen/selected_plate
    2

However there will also be ``model_evaluation.h5`` files for each iteration which save how
the model performed on the holdout set given training on the plates revealed up until that point.
These can be collected for analysis after the simulation completes.


Computational Cost
------------------

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
