include { RETROSPECTIVE } from './nextflow/workflows/nf-core/batchie/retrospective_simulation/main.nf'
include { PROSPECTIVE } from './nextflow/workflows/nf-core/batchie/prospective/main.nf'
include { NEXT_BATCH_PLATE } from './nextflow/workflows/nf-core/batchie/next_batch_plate/main.nf'

workflow {
    if (params.mode == 'retrospective') {
        RETROSPECTIVE (  )
    }

    if (params.mode == 'prospective') {
        PROSPECTIVE (  )
    }

    if (params.mode == 'next_plate') {
        NEXT_BATCH_PLATE (  )
    }
}
