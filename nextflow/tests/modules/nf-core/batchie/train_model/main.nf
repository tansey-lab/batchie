#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { TRAIN_MODEL } from '../../../../../modules/nf-core/batchie/train_model/main'

workflow train_model {
    input = [ [ id:'test', single_end:false ], // meta map
              file(params.test_data['batchie']['unmasked_dataset'], checkIfExists: true),
              0, // chain_index
              1 // n_chains
            ]
    TRAIN_MODEL ( input )
}
