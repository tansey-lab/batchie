#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { REVEAL_PLATE } from '../../../../../modules/nf-core/batchie/reveal_plate/main'

workflow reveal_plate {

    // tuple val(meta), path(data_masked), path(data_unmasked), path(thetas), path(batch_selection), path(simulation_tracker)
    input = [ [ id:'test', single_end:false ], // meta map
              file(params.test_data['batchie']['masked_screen'], checkIfExists: true),
              [1,2]
            ]
    REVEAL_PLATE ( input )
}
