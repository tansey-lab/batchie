include { RETROSPECTIVE } from './nextflow/workflows/nf-core/batchie/retrospective_simulation/main.nf'
include { PROSPECTIVE } from './nextflow/workflows/nf-core/batchie/prospective/main.nf'

workflow {
    if (params.mode == 'retrospective') {
        RETROSPECTIVE (  )
    }

    if (params.mode == 'prospective') {
        PROSPECTIVE (  )
    }
}
