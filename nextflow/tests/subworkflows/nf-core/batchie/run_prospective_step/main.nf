#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { RUN_PROSPECTIVE_STEP } from '../../../../../subworkflows/nf-core/batchie/run_prospective_step/main'

workflow run_prospective_step {
    // channel: [ val(meta), path(experiment), val(n_chains), val(n_chunks) ]
    input = tuple( [ id:'test', single_end:false ], // meta map
              file(params.test_data['batchie']['masked_experiment'], checkIfExists: true),
              3,
              3
            )
    ch_input = Channel.fromList([input])
    RUN_PROSPECTIVE_STEP ( ch_input )
}
