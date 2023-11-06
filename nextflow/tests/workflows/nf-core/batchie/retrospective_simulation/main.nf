#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { RETROSPECTIVE_SIMULATION } from '../../../../../workflows/nf-core/batchie/retrospective_simulation/main'

workflow retrospective_simulation_test {
    RETROSPECTIVE_SIMULATION (  )
}
