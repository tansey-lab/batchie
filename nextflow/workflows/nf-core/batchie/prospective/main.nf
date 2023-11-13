#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { RUN_PROSPECTIVE_STEP } from '../../../../subworkflows/nf-core/batchie/run_prospective_step/main'

workflow PROSPECTIVE {
    def name = params.name == null ? "batchie" : params.name

    // channel: [ val(meta), path(screen), val(n_chains), val(n_chunks) ]
    input = tuple( [ id: name, single_end:false ], // meta map
              file(params.screen, checkIfExists: true),
              params.n_chains,
              params.n_chunks
            )
    ch_input = Channel.fromList([input])
    RUN_PROSPECTIVE_STEP ( ch_input )
}
