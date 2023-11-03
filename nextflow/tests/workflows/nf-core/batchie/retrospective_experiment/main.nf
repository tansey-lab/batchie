#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { RETROSPECTIVE_EXPERIMENT } from '../../../../../workflows/nf-core/batchie/retrospective_experiment/main'

workflow retrospective_experiment_test {
    RETROSPECTIVE_EXPERIMENT (  )
}
