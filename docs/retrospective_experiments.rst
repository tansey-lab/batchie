Retrospective Experiments
=========================

Below are commands to reproduce our retrospective experiments.

The input ``.screen.h5`` files are available for download here: https://zenodo.org/records/12764821

Warning: These retrospective simulation experiments require a lot of computation time. We recommend these be
run on a high performance computing cluster or cloud computing service. Use the appropriate
`-profile` to specify the computing environment.


Merck
-----

..code-block:: bash

    python3 nextflow/scripts/batchie.py --mode retrospective \
        -c nextflow/config/merck_retrospective_simulation.config \
        --batch-size 9 \
        --screen merck_2016.screen.h5 \
        --outdir merck_sim \
        --max_memory 64G \
        --max_cpus 1 \
        --max_time 24h


NCI Almanac
-----------

..code-block:: bash

    python3 nextflow/scripts/batchie.py --mode retrospective \
        -c nextflow/config/nci_almanac_retrospective_simulation.config \
        --batch-size 18 \
        --screen nci_almanac.screen.h5 \
        --outdir merck_sim \
        --max_memory 64G \
        --max_cpus 1 \
        --max_time 24h


GDSC^2
------

..code-block:: bash

    python3 nextflow/scripts/batchie.py --mode retrospective \
        -c nextflow/config/gdsc_sq_retrospective_simulation.config \
        --batch-size 36 \
        --screen gdsc_sq.screen.h5 \
        --outdir merck_sim \
        --max_memory 64G \
        --max_cpus 1 \
        --max_time 24h