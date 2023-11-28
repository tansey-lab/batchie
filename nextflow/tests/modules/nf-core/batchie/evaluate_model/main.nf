#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { EVALUATE_MODEL } from '../../../../../modules/nf-core/batchie/evaluate_model/main'

workflow evaluate_model {
    input = [ [ id:'test', single_end:false ], // meta map
              file(params.test_data['batchie']['masked_screen'], checkIfExists: true),
              file(params.test_data['batchie']['unmasked_screen'], checkIfExists: true),
              file(params.test_data['batchie']['thetas'], checkIfExists: true)
            ]
    EVALUATE_MODEL ( input )
}
