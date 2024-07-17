.. _install:

Installation
============

Prerequisites
-------------

Nextflow and Python (>=3.11) are required to run the BATCHIE pipeline. The ``nextflow`` and ``python3`` commands must be available.

Instructions for installing nextflow are here (installation time is ~1 minute):
https://www.nextflow.io/docs/latest/install.html

If you want to use the containerized version of BATCHIE, installing docker is necessary.
Instructions for installing docker are here: https://docs.docker.com/get-docker/

Option 1: Install Using pip
---------------------------

.. code-block:: sh

    python3 -m venv venv
    source venv/bin/activate
    pip install git+https://github.com/tansey-lab/batchie

On the authors an Apple M1 13" MacBook Pro (2020) with 16GB of RAM, execution of the above command took 23.114 seconds.

Option 2: Using Docker
----------------------

The most reproducible way to run BATCHIE is to use our docker container.

We maintain up-to-date and historical docker images on docker hub, you can pull the latest version of BATCHIE with the following command:

.. code-block:: sh

    docker pull jeffquinnmsk/batchie:latest


Depending on internet connectivity, this should not take longer than a few minutes.
The compressed size of the BATCHIE docker image is 5.94 GB at the time of writing.