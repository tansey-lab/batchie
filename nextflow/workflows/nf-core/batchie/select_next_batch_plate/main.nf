#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { SELECT_NEXT_BATCH_PLATE } from '../../../../subworkflows/nf-core/batchie/select_next_batch_plate/main'

workflow SELECT_NEXT_BATCH_PLATE {
    def name = params.name == null ? "batchie" : params.name

    // channel: [ val(meta), path(screen), path(thetas), path(distance_matrix_chunks), val(n_chunks), val(reveal) ]
    input = tuple( [ id: name, single_end:false ], // meta map
              file(params.screen, checkIfExists: true),
              file(params.thetas, checkIfExists: true),
              file(params.distance_matrix, checkIfExists: true),
              params.n_chunks,
              params.reveal
            )
    ch_input = Channel.fromList([input])
    SELECT_NEXT_BATCH_PLATE ( ch_input )
}
