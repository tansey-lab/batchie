.. _data-format:

Batchie Data Format
===================

The main data format for batchie is the :py:class:`batchie.data.Screen` object.

Example
-------

See below for an example of creating a screen object and saving it to disk:

.. code-block:: python

    from batchie.data import Screen

    screen = Screen(
        observations=np.array([0.1, 0.2, 0.0, 0.0]),
        observation_mask=np.array([True, True, False, False]),
        sample_names=np.array([
            "sample_A",
            "sample_B",
            "sample_C",
            "sample_D"
        ], dtype=str),
        plate_names=np.array([
            "plate_A",
            "plate_A",
            "plate_B",
            "plate_B"
        ], dtype=str),
        treatment_names=np.array(
            [
                ["drug_A", "drug_B"],
                ["drug_A", "drug_B"],
                ["drug_A", "drug_B"],
                ["drug_A", "control"]
            ], dtype=str
        ),
        treatment_doses=np.array(
            [
                [2.0, 2.0],
                [1.0, 2.0],
                [2.0, 1.0],
                [2.0, 0]
            ]
        ),
        control_treatment_name="control"
    )

    screen.save_h5("my_screen.screen.h5")


Explanation of Terms
--------------------

A ``Screen`` contains one or more experiments. Each experiment has a set of conditions,
the conditions are the ``treatment_names`` (drug names for instance) and
the ``treatment_doses`` (drug concentration for instance). If some treatments are considered "controls"
or to have no effect, they should be equal to the indicated ``control_treatment_name``.

Each experiment is on a particular sample, indicated by the ``sample_names``.

Each experiment has an optional observation, passed via the ``observations`` parameter.

The observation is the outcome of observing the experiment, a single floating point number.
In the case of high throughput drug screening, this is the viability of the cells.
If the experiment has no observation yet, then you can indicate that by passing an ``observation_mask``
with ``False`` values for the experiments that have no observation yet.

Each experiment belongs to a particular plate, passed via the ``plate_names`` parameter.

You create a batchie screen object in python by passing it numpy arrays for all of these various parameters.
All the numpy arrays you pass in are expected to be the same length.
The numpy arrays are also expected to be aligned, aka index N of each array
corresponds to the same experiment for all N in ``range(screen.size)``

Furthermore ``treatment_names`` and ``treatment_doses`` are expected to be the same width (just set the dose to 0 for controls).
All of these things are sanity checked so if you make a mistake, initializing the screen will raise an exception.

Assuming you are comfortable working with numpy, it should be very straightforward to create a screen object from your dataset.

Explanation of Example
----------------------

In this example in the code block above we have a screen that is looking at 4 experiments, with arity of 2 (looking at combinations of
2 treatments per experiment). There are two unique plates, with two experiments each. On ``plate_B`` the second experiment
has a control, so we are only measuring the effect of ``drug_A`` in that experiment.

The ``observation_mask`` indicates that the experiments in ``plate_A`` have been observed, but the experiments in ``plate_B``
have not (the value of the ``observations`` array for experiments where ``observation_mask`` is ``False`` is ignored,
you can just set it to zero or any value).

On-Disk Format
--------------

The on-disk format for the screen is a compressed HDF5 file.
We use the extension `.screen.h5` for these files by convention. Once you have your `.screen.h5` file prepared,
you are ready to start running batchie!
