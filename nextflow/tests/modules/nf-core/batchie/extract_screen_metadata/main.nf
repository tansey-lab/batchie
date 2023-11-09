#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { EXTRACT_SCREEN_METADATA } from '../../../../../modules/nf-core/batchie/extract_screen_metadata/main'

workflow extract_screen_metadata {
    input = [ [ id:'test', single_end:false ], // meta map
              file(params.test_data['batchie']['masked_screen'], checkIfExists: true)
            ]
    EXTRACT_SCREEN_METADATA ( input )
}
