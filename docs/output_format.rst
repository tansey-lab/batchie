.. _data-format:

BATCHIE Output Format
=====================

The output of the BATCHIE pipeline is a two-level hierarchical directory structure.
The top level corresponds to the "rounds" or iterations of the batchie algorithm, and the level below that corresponds
to the "batch" or number of plates revealed in each round.

For example, the output of the demo retrospective pipeline will look like this:

.. code-block:: text

    retrospective_sim
    ├── iter_0
    │   ├── plate_0
    │   │   └── unmasked_screen
    │   │       ├── advanced_screen.h5 
    │   │       ├── distance_matrix_chunk_0.h5 
    │   │       ├── distance_matrix_chunk_1.h5 
    │   │       ├── model_evaluation.h5 
    │   │       ├── model_evaluation_analysis 
    │   │       ├── score_chunk_0.h5 
    │   │       ├── score_chunk_1.h5 
    │   │       ├── screen_metadata.json 
    │   │       ├── selected_plate 
    │   │       ├── test.screen.h5 
    │   │       ├── thetas_0.h5 
    │   │       ├── thetas_1.h5 
    │   │       └── training.screen.h5 
    │   ├── plate_1
    │   │   └── unmasked_screen
    │   │       ├── advanced_screen.h5 
    │   │       ├── score_chunk_0.h5 
    │   │       ├── score_chunk_1.h5 
    │   │       ├── screen_metadata.json 
    │   │       └── selected_plate 
    │   └── plate_2
    │       └── unmasked_screen
    │           ├── advanced_screen.h5 
    │           ├── score_chunk_0.h5 
    │           ├── score_chunk_1.h5 
    │           ├── screen_metadata.json 
    │           └── selected_plate 
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

We ran this demo with a ``--batch-size`` parameter of 3, this means three plates will be selected in each round (up to the point at which there
are no plates left to reveal). Our demo dataset has 5 plates total, one of which is revealed at the start of the simulation, 
and 4 of which will be chosen by the model. 
As a result the model runs for two rounds, in the first round it chooses 3 plates, and in the second round it chooses the final
plate, and finally it exits as there are no more plates to reveal.

We see there are two directories at the top level of ``retrospective_sim``, ``iter_0`` and ``iter_1``, corresponding to each round.
In ``iter_0`` we see three directories, ``plate_0``, ``plate_1``, and ``plate_2``, corresponding to the three plates chosen in the first round.
In ``iter_1`` we see one directory, ``plate_0``, corresponding to the one plate chosen in the second round.

Before the first plate selection in each round, the underlying model is trained, and then pairwise distance calculations are performed
between the predictions over all the remaining unobserved plates. When the batch size is greater than 1, we do not retrain the model for the
subsequent plates in each round. This is why the number and composition of output files is different for the ``plate_0`` directory.

Glossary of Output Files
------------------------

``advanced_screen.h5``
^^^^^^^^^^^^^^^^^^^^^^

A copy of the screen object with the current plate revealed (only relevant for retrospective mode).

``distance_matrix_chunk_<0..n_chunks>.h5``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The pairwise distance matrix between the predictions of the model on the unobserved plates.

Because the pairwise distance matrix can be extremely large, we calculate it in chunks in parallel.

The full distance matrix can be read in with the following code:

.. code-block:: python

    import glob
    from batchie.distance_calculation import ChunkedDistanceMatrix

    distance_matrix = ChunkedDistanceMatrix.concat(
            [ChunkedDistanceMatrix.load(fn) for fn in glob.glob("distance_matrix_chunk_*.h5")]
        )
    M = distance_matrix.to_dense()


``model_evaluation.h5``
^^^^^^^^^^^^^^^^^^^^^^^

Evaluation of the model on the holdout set (only relevant for retrospective mode).

You can inspect this file with the following code:

.. code-block:: python

    from batchie.models.main import ModelEvaluation

    model_evaluation = ModelEvaluation.load_h5("model_evaluation.h5")
    model_evaluation.mse()


``model_evaluation_analysis``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This folder contains various plots showing the performance of the model on the holdout set (only relevant for retrospective mode).

``score_chunk_<0..n_chunks>.h5``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Plate scoring via PDBAL is computationally expensive, so we parallelize the scoring by doing one chunk of plates in each process.
Each of these files contains the scores for one chunk of plates.

The scores for all plates can be read in with the following code:

.. code-block:: python

    import glob
    from batchie.scoring.main import ChunkedScoresHolder

    scores = ChunkedScoresHolder.concat(
            [ChunkedScoresHolder.load_h5(fn) for fn in glob.glob("score_chunk_*.h5")]
        )
    scores.get_score(plate_id)

``thetas_<0..n_chains>.h5``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

These objects contained the learned parameters for the model. In each round we train ``--n_chains`` models, each with a different random seed. 
One ``ThetaHolder`` object is saved for each trained model.

You can read in all the learned parameters with the following code:

.. code-block:: python

    import glob
    from batchie.core import ThetaHolder

    theta_holder = ThetaHolder(n_thetas=len(glob.glob("thetas_*.h5")))

    theta_holders = [theta_holder.load_h5(fn) for fn in glob.glob("thetas_*.h5")]

    thetas = theta_holder.concat(theta_holders)
    thetas.get_theta(0)

The actual Theta object itself that is returned by ``get_theta`` is implementation specific. Different models have different parameter structures,
so when you implement a new model in the BATCHIE framework you'll need to also implement Theta.

``screen_metadata.json``
^^^^^^^^^^^^^^^^^^^^^^^^

Metadata about the current state of the screen, for example:

.. code-block:: json

    {
        "n_unique_samples": 2,
        "n_unique_treatments": 2,
        "size": 8,
        "n_plates": 5,
        "n_unobserved_plates": 0,
        "n_observed_plates": 5
    }

``selected_plate``
^^^^^^^^^^^^^^^^^^

A text file with the name of the plate that was selected in this step:

.. code-block:: shell

    $ cat retrospective_sim/iter_1/plate_0/unmasked_screen/selected_plate
    3

``test.screen.h5``
^^^^^^^^^^^^^^^^^^

A subset of the Screen which will be held out for model evaluation (only relevant for retrospective mode).

``training.screen.h5``
^^^^^^^^^^^^^^^^^^^^^^

A subset of the Screen which will be used for training the model.
