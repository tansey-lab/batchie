#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { RUN_RETROSPECTIVE_STEP } from '../../../../../subworkflows/nf-core/batchie/run_retrospective_step/main'

workflow run_retrospective_step {
    // channel: [ val(meta), path(masked_screen), path(unmasked_screen), path(simulation_tracker), val(n_chains), val(n_chunks) ]
    input = tuple( [ id:'test', single_end:false ], // meta map
              file(params.test_data['batchie']['masked_screen'], checkIfExists: true),
              file(params.test_data['batchie']['unmasked_screen'], checkIfExists: true),
              file(params.test_data['batchie']['simulation_tracker'], checkIfExists: true),
              3,
              3
            )
    ch_input = Channel.fromList([input])
    RUN_RETROSPECTIVE_STEP ( ch_input )
}
