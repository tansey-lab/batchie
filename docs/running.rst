Running BATCHIE
===============

After you have created your batchie :py:class:`batchie.data.Screen` file (see here: :ref:`data-format`),
you can run batchie using our configurable nextflow pipelines.

There are two main modes for running batchie, :ref:`running-prospective` and :ref:`running-retrospective`.

.. _running-prospective:

Prospective
-----------

Assume you have a :py:class:`batchie.data.Screen` containing one or more
:py:class:`batchie.data.Plate` s that you have not yet observed.

In prospective mode, the batchie model will tell you which of those
unobserved plates should be run next.

.. code-block:: bash

    nextflow run nextflow/workflows/nf-core/batchie/prospective/main.nf -c ./nextflow/config/ \
        --profile docker \
        --screen <path to screen.json> \
        --n_chains <number of model chains to run>
        --n_chunks <number of chunks to split the data into>


.. _running-retrospective:

Retrospective
-------------

Assume you have a :py:class:`batchie.data.Screen` that has already been run, such that
all :py:class:`batchie.data.Plate` s are observed.

In retrospective mode, the batchie model will first mask all of these plates.

We will sample a fraction of the screen and set it aside as "holdout" for evaluating model accuracy.

Then we will pick one initial plate to reveal. For the initial plate reveal,
one has the option of using a configured heuristic algorithm or picking an existing plate at random.

Once the initial plate is revealed, we repeat the following steps until all plates have been revealed:

#. Train the specified model given the revealed plates
#. Use the trained model to predict on all unobserved plates
#. Use the model predictions on unobserved data to score all the unobserved plates
#. Chose one or more plates with the best scores to reveal
