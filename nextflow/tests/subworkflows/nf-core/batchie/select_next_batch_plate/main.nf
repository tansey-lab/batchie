#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { SELECT_NEXT_BATCH_PLATE } from '../../../../../subworkflows/nf-core/batchie/select_next_batch_plate/main'

workflow select_next_batch_plate {
    // channel: [ val(meta), path(masked_screen), path(unmasked_screen), val(n_chains), val(n_chunks) ]
    input = tuple( [ id:'test', single_end:false ], // meta map
              file(params.test_data['batchie']['masked_screen'], checkIfExists: true),
              file(params.test_data['batchie']['thetas'], checkIfExists: true),
              file(params.test_data['batchie']['distance_matrix'], checkIfExists: true),
              3,
              true,
              []
            )
    ch_input = Channel.fromList([input])
    SELECT_NEXT_BATCH_PLATE ( ch_input )
}
