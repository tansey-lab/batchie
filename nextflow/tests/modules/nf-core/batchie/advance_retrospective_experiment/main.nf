#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { ADVANCE_RETROSPECTIVE_SIMULATION } from '../../../../../modules/nf-core/batchie/advance_retrospective_simulation/main'

workflow advance_retrospective_simulation {

    // tuple val(meta), path(data_masked), path(data_unmasked), path(thetas), path(batch_selection), path(experiment_tracker)
    input = [ [ id:'test', single_end:false ], // meta map
              file(params.test_data['batchie']['masked_experiment'], checkIfExists: true),
              file(params.test_data['batchie']['unmasked_experiment'], checkIfExists: true),
              file(params.test_data['batchie']['thetas'], checkIfExists: true),
              file(params.test_data['batchie']['batch_selection'], checkIfExists: true),
              file(params.test_data['batchie']['experiment_tracker'], checkIfExists: true),
            ]
    ADVANCE_RETROSPECTIVE_SIMULATION ( input )
}
