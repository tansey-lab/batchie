#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { SELECT_NEXT_BATCH_PLATE } from '../../../../subworkflows/nf-core/batchie/select_next_batch_plate/main'
include { EXTRACT_SCREEN_METADATA } from '../../../../modules/nf-core/batchie/extract_screen_metadata/main'

process WRITE_REMAINING_PLATES {
    tag "$meta.id"
    label 'process_single'

    input:
    tuple val(meta), val(n_unobserved_plates)

    output:
    tuple val(meta), path("${prefix}/n_remaining_plates"), emit: n_remaining_plates

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    mkdir -p "${prefix}"
    echo "${n_unobserved_plates}" > "${prefix}/n_remaining_plates"
    """
}

workflow NEXT_BATCH_PLATE {
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

    EXTRACT_SCREEN_METADATA( RUN_RETROSPECTIVE_STEP.out.ch_output.map { tuple(it[0], it[1]) } )
}
