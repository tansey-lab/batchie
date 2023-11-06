#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { CALCULATE_DISTANCE_MATRIX_CHUNK } from '../../../../../modules/nf-core/batchie/calculate_distance_matrix_chunk/main'

workflow calculate_distance_matrix_chunk {

    //     tuple val(meta), path(data), path(thetas), val(chunk_index), val(n_chunks)
    input = [ [ id:'test', single_end:false ], // meta map
              file(params.test_data['batchie']['masked_screen'], checkIfExists: true),
              file(params.test_data['batchie']['thetas'], checkIfExists: true),
              0,
              1
            ]
    CALCULATE_DISTANCE_MATRIX_CHUNK ( input )
}
