#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { SELECT_NEXT_PLATE } from '../../../../../modules/nf-core/batchie/select_next_plate/main'

workflow select_next_plate {
    input = [ [ id:'test', single_end:false ], // meta map
              file(params.test_data['batchie']['masked_screen'], checkIfExists: true),
              [11, 12],
              file(params.test_data['batchie']['scores'], checkIfExists: true)
            ]
    SELECT_NEXT_PLATE ( input )
}
