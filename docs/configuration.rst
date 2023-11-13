Configuring BATCHIE
===================

BATCHIE is a pipeline made up of several command line interface applications.

Documentation for each individual command line application is available here: :ref:`cli`. All of these applications
can be used individually, but they are intended to be used together in a
nextflow pipeline.

The entire pipeline is configured using a single nextflow ``.config`` file. Several pre-defined configuration files are
available in `nextflow/config <https://github.com/tansey-lab/batchie/tree/main/nextflow/config>`_ . We recommend using
these as templates for your own configuration files.

Let's take a look at an example configuration file to explain its contents:

.. code-block:: groovy

    params {
        n_chunks = 1
        n_chains = 1
    }

    process {
        publishDir = { "${params.outdir}" }

        withName: TRAIN_MODEL {
            ext.args = [
                '--model', 'SparseDrugCombo',
                '--model-param', 'n_embedding_dimensions=10',
                '--n-burnin', '10',
                '--n-samples', '10',
                '--thin', '1'
            ].join(' ')
        }

        withName: CALCULATE_DISTANCE_MATRIX_CHUNK {
            ext.args = [
                '--model', 'SparseDrugCombo',
                '--model-param', 'n_embedding_dimensions=10',
                '--distance-metric', 'MSEDistance'
            ].join(' ')
        }

        withName: SELECT_NEXT_BATCH {
            ext.args = [
                '--model', 'SparseDrugCombo',
                '--model-param', 'n_embedding_dimensions=10',
                '--scorer', 'GaussianDBALScorer',
                '--batch-size', '1'
            ].join(' ')
        }
    }

The first section of the configuration file is the ``params`` section. This section defines pipeline-wide parameters.
``n_chunks`` is the number of chunks to split
distance matrix calculation into and ``n_chains`` is the number of parallel MCMC chains to run.

Pipeline wide parameters can be passed to nextflow on the command line using the format ``--<param_name> <param_value>``.
You'll probably want to pass ``--outdir`` and ``--screen`` on the command line instead of hardcoding them in
the config file, but it's up to you.

After that we have the process-wide section. The line:

.. code-block:: groovy

    publishDir = { "${params.outdir}" }

Tells nextflow to publish the output of all processes to the directory specified by the ``outdir`` parameter.

And following that we have process-specific sections. This section:

.. code-block:: groovy

    withName: TRAIN_MODEL {
        ext.args = [
            '--model', 'SparseDrugCombo',
            '--model-param', 'n_embedding_dimensions=10',
            '--n-burnin', '10',
            '--n-samples', '10',
            '--thin', '1'
        ].join(' ')
    }

Specifies what command line options will be passed to the call to :ref:`cli_train_model`. You'll define
a similar section for each other application which you want to pass command line options to. Some applications
dont have any required command line options, so you can leave those sections out entirely.
