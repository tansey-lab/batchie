include { PREPARE_RETROSPECTIVE_EXPERIMENT } from '../../../../modules/nf-core/batchie/prepare_retrospective_experiment/main'
include { EXTRACT_EXPERIMENT_METADATA } from '../../../../modules/nf-core/batchie/extract_experiment_metadata/main'
include { RUN_RETROSPECTIVE_STEP } from '../../../../subworkflows/nf-core/batchie/run_retrospective_step/main'


process WRITE_REMAINING_PLATES {
    tag "$meta.id"
    label 'process_single'

    input:
    tuple val(meta), val(n_unobserved_plates)

    output:
    tuple val(meta), path("${prefix}/n_remaining_plates"), emit: n_remaining_plates

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    mkdir -p "${prefix}"
    echo "${n_unobserved_plates}" > "${prefix}/n_remaining_plates"
    """
}

workflow RETROSPECTIVE_EXPERIMENT {
    def prepared_input = null

    if (params.masked_experiment == null || params.experiment_tracker == null) {
        def input_tuple = tuple([id:params.experiment_name], file(params.unmasked_experiment))
        def input_value_channel = Channel.fromList( [input_tuple] )
        PREPARE_RETROSPECTIVE_EXPERIMENT( input_value_channel )

        prepared_input = PREPARE_RETROSPECTIVE_EXPERIMENT.out.experiment
            .join(input_value_channel)
            .map { tuple(it[0], it[1], it[2], [], params.n_chains, params.n_chunks) }
            .collect()
    } else {
        def intermediate_input_tuple = tuple(
            [id:params.experiment_name],
            file(params.masked_experiment),
            file(params.unmasked_experiment),
            file(params.experiment_tracker),
            params.n_chains,
            params.n_chunks
            )
        prepared_input = Channel.fromList( [input_tuple] )
    }

    RUN_RETROSPECTIVE_STEP( prepared_input )

    EXTRACT_EXPERIMENT_METADATA( RUN_RETROSPECTIVE_STEP.out.ch_output.map { tuple(it[0], it[1]) } )

    WRITE_REMAINING_PLATES( EXTRACT_EXPERIMENT_METADATA.out.n_unobserved_plates )
}
