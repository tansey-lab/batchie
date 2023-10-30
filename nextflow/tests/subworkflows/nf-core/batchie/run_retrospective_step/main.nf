#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { RUN_RETROSPECTIVE_STEP } from '../../../../../subworkflows/nf-core/batchie/run_retrospective_step/main'

workflow run_retrospective_step {
    // channel: [ val(meta), path(masked_experiment), path(unmasked_experiment), path(experiment_tracker), val(n_chains), val(n_chunks), val(is_complete) ]
    input = tuple( [ id:'test', single_end:false ], // meta map
              file(params.test_data['batchie']['masked_experiment'], checkIfExists: true),
              file(params.test_data['batchie']['unmasked_experiment'], checkIfExists: true),
              file(params.test_data['batchie']['experiment_tracker'], checkIfExists: true),
              3,
              3
            )
    ch_input = Channel.fromList([input])
    RUN_RETROSPECTIVE_STEP ( ch_input )
}
