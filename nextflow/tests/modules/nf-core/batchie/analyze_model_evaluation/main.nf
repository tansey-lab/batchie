#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { ANALYZE_MODEL_EVALUATION } from '../../../../../modules/nf-core/batchie/analyze_model_evaluation/main'

workflow evaluate_model {
    input = [ [ id:'test', single_end:false ], // meta map
              file(params.test_data['batchie']['masked_screen'], checkIfExists: true),
              file(params.test_data['batchie']['thetas'], checkIfExists: true),
              file(params.test_data['batchie']['model_evaluation'], checkIfExists: true),
            ]
    ANALYZE_MODEL_EVALUATION ( input )
}
