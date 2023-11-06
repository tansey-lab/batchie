#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { PREPARE_RETROSPECTIVE_SIMULATION } from '../../../../../modules/nf-core/batchie/prepare_retrospective_simulation/main'

workflow prepare_retrospective_simulation {

    // tuple val(meta), path(data_masked), path(data_unmasked), path(thetas), path(batch_selection), path(experiment_tracker)
    input = [ [ id:'test', single_end:false ], // meta map
              file(params.test_data['batchie']['unmasked_experiment'], checkIfExists: true),
            ]
    PREPARE_RETROSPECTIVE_SIMULATION ( input )
}
