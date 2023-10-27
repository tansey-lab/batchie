
include { PREPARE_RETROSPECTIVE_EXPERIMENT } from '../../../../modules/nf-core/batchie/prepare_retrospective_experiment/main'
include { EXTRACT_EXPERIMENT_METADATA } from '../../../../modules/nf-core/batchie/extract_experiment_metadata/main'
include { RUN_RETROSPECTIVE_STEP } from '../../../../subworkflows/nf-core/batchie/run_retrospective_step/main'

workflow RETROSPECTIVE_EXPERIMENT {
    input = tuple([id:params.experiment_name], file(params.experiment_file))

    PREPARE_RETROSPECTIVE_EXPERIMENT( Channel.fromList( [input] ) )

    prepared_input = PREPARE_RETROSPECTIVE_EXPERIMENT.out.retrospective_experiment.collect()[0]

    EXTRACT_EXPERIMENT_METADATA( PREPARE_RETROSPECTIVE_EXPERIMENT.out.retrospective_experiment )

    n_unobserved_plates = EXTRACT_EXPERIMENT_METADATA.out.n_unobserved_plates.collect()[0][1]

    n_iterations = ceil(n_unobserved_plates / params.batch_size)

    log.info("Running retrospective experiment for ${n_iterations} iterations")

/*
    RUN_RETROSPECTIVE_STEP
        .recurse(prepared_input)
        .times(n_iterations)
        */
}
