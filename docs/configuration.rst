Configuring BATCHIE
===================

Pipeline Wide Parameters
------------------------

Pipeline-wide parameters affect the execution of the pipeline as a whole. These parameters can be specificed
on the command line when running ``batchie.py``.

``--n_chunks`` is the number of chunks to split distance matrix/plate score calculation into.
``--n_chains`` is the number of parallel MCMC chains to run.
``--batch-size`` is the number of plates to select per iteration
``--outdir`` is the output directory
``--screen`` is the screen to use for running the pipeline


Process Specific Parameters
---------------------------

BATCHIE as a pipeline is made up of several component Python command line interface applications.

Documentation for each individual command line application is available here: :ref:`cli`. Command line parameters
can be passed into each application using a single nextflow ``.config`` file.

Several pre-defined configuration files are
available in `nextflow/config <https://github.com/tansey-lab/batchie/tree/main/nextflow/config>`_ . We recommend using
these as templates for your own configuration files.

Let's take a look at an example configuration file to explain its contents:

.. code-block:: groovy

    process {
        publishDir = { "${params.outdir}" }

        withName: TRAIN_MODEL {
            ext.args = [
                '--model', 'LegacySparseDrugCombo',
                '--model-param', 'n_embedding_dimensions=10',
                '--n-burnin', '10',
                '--n-samples', '10',
                '--thin', '1'
            ].join(' ')
        }

        withName: CALCULATE_DISTANCE_MATRIX_CHUNK {
            ext.args = [
                '--model', 'LegacySparseDrugCombo',
                '--model-param', 'n_embedding_dimensions=10',
                '--distance-metric', 'MSEDistance'
            ].join(' ')
        }

        withName: CALCULATE_SCORE_CHUNK {
            ext.args = [
                '--model', 'LegacySparseDrugCombo',
                '--model-param', 'n_embedding_dimensions=10',
                '--scorer', 'GaussianDBALScorer',
                '--seed', '12'
            ].join(' ')
        }
    }


This is boilerplate that tells nextflow that the output for all processes should be published to the directory
specified in the pipeline-wide ``--outdir`` parameter:

.. code-block:: groovy

    process {
        publishDir = { "${params.outdir}" }


This section:

.. code-block:: groovy

    withName: TRAIN_MODEL {
        ext.args = [
            '--model', 'LegacySparseDrugCombo',
            '--model-param', 'n_embedding_dimensions=10',
            '--n-burnin', '10',
            '--n-samples', '10',
            '--thin', '1'
        ].join(' ')
    }

Specifies what command line options will be passed to the call to :ref:`cli_train_model`. You'll define
a similar section for each other application which you want to pass command line options to. Some applications
dont have any required command line options, so you can leave those sections out entirely.

You can always refer
to the :ref:`cli` documentation for more information on what options are available for each process.
