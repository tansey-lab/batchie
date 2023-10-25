#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { SELECT_NEXT_BATCH } from '../../../../../modules/nf-core/batchie/select_next_batch/main'

workflow select_next_batch {
    input = [ [ id:'test', single_end:false ], // meta map
              file(params.test_data['batchie']['masked_experiment'], checkIfExists: true),
              file(params.test_data['batchie']['thetas'], checkIfExists: true),
              file(params.test_data['batchie']['distance_matrix'], checkIfExists: true)
            ]
    SELECT_NEXT_BATCH ( input )
}
