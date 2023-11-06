Installation
============

This tutorial will walk you through the process of setting up an environment
to run Batchie.

Option 1 (Recommended): Using Docker
------------------------------------

Batchie uses several python packages with C extensions,
so the easiest way to get started is using the up to date
docker image we maintain on docker hub.

.. code::

    docker pull jeffquinnmsk/batchie:latest

Option 2: Install Using pip
---------------------------

Batchie can be installed directly as a python package using pip.

.. code::

    pip install git+https://github.com/tansey-lab/bayestme
