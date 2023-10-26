#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { EXTRACT_EXPERIMENT_METADATA } from '../../../../../modules/nf-core/batchie/extract_experiment_metadata/main'

workflow extract_experiment_metadata {
    input = [ [ id:'test', single_end:false ], // meta map
              file(params.test_data['batchie']['masked_experiment'], checkIfExists: true)
            ]
    EXTRACT_EXPERIMENT_METADATA ( input )
}
