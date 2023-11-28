#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { CALCULATE_SCORE_CHUNK } from '../../../../../modules/nf-core/batchie/calculate_score_chunk/main'

workflow calculate_score_chunk {
    input = [ [ id:'test', single_end:false ], // meta map
              file(params.test_data['batchie']['masked_screen'], checkIfExists: true),
              file(params.test_data['batchie']['thetas'], checkIfExists: true),
              file(params.test_data['batchie']['distance_matrix'], checkIfExists: true),
              [1],
              0,
              1
            ]
    CALCULATE_SCORE_CHUNK ( input )
}
