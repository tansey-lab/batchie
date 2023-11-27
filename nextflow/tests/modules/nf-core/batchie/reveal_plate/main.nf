#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { ADVANCE_RETROSPECTIVE_SIMULATION } from '../../../../../modules/nf-core/batchie/advance_retrospective_simulation/main'

workflow advance_retrospective_simulation {

    // tuple val(meta), path(data_masked), path(data_unmasked), path(thetas), path(batch_selection), path(simulation_tracker)
    input = [ [ id:'test', single_end:false ], // meta map
              file(params.test_data['batchie']['masked_screen'], checkIfExists: true),
              [1,2]
            ]
    ADVANCE_RETROSPECTIVE_SIMULATION ( input )
}
