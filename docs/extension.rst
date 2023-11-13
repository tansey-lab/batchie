Extending BATCHIE
=================

BATCHIE is designed to be extended with new models and methods.

Adding a new model
------------------

To add a new model for prediction of outcomes, you will need to create an implementation of the
:py:class:`batchie.core.BayesianModel` class. You will also need to implement the
:py:class:`batchie.core.Theta` and :py:class:`batchie.core.ThetaHolder` class for working
with and serializing the model parameters.

The command line applications where the model is used (for instance
:ref:`cli_train_model`) accept ``--model`` and ``--model-params`` arguments.
``--model`` is the name of the class implementing the model, and ``--model-params``
are arbitrary key-value pairs that will be passed to the model constructor (``__init__`` method).

Other Extensible Classes
------------------------

Other than the model, the :py:class:`batchie.core.Scorer`, :py:class:`batchie.core.DistanceMetric`, and
:py:class:`batchie.core.PlatePolicy` can also be implemented and specified in the command line applications
in the same way.

For retrospective simulations :py:class:`batchie.core.RetrospectivePlateGenerator`,
:py:class:`batchie.core.InitialRetrospectivePlateGenerator`, and :py:class:`batchie.core.RetrospectivePlateSmoother`,
can also be implemented and specified for the :ref:`cli_prepare_retrospective_simulation` command line application.
