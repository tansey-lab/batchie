include { PREPARE_RETROSPECTIVE_SIMULATION } from '../../../../modules/nf-core/batchie/prepare_retrospective_simulation/main'
include { EXTRACT_SCREEN_METADATA } from '../../../../modules/nf-core/batchie/extract_screen_metadata/main'
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

workflow RETROSPECTIVE {
    def name = params.name == null ? "batchie" : params.name

    def prepared_input = null

    if (params.initialize == true) {
        def input_tuple = tuple([id:name], file(params.screen))
        def input_value_channel = Channel.fromList( [input_tuple] )
        PREPARE_RETROSPECTIVE_SIMULATION( input_value_channel )

        prepared_input = PREPARE_RETROSPECTIVE_SIMULATION.out.training_screen
            .join(PREPARE_RETROSPECTIVE_SIMULATION.out.test_screen)
            .map { tuple(it[0], it[1], it[2], params.n_chains, params.n_chunks) }
            .collect()
    } else {
        def intermediate_input_tuple = tuple(
            [id:params.simulation_name],
            file(params.training_screen),
            file(params.test_screen),
            params.n_chains,
            params.n_chunks
            )
        prepared_input = Channel.fromList( [intermediate_input_tuple] )
    }

    RUN_RETROSPECTIVE_STEP( prepared_input )

    EXTRACT_SCREEN_METADATA( RUN_RETROSPECTIVE_STEP.out.ch_output.map { tuple(it[0], it[1]) } )

    WRITE_REMAINING_PLATES( EXTRACT_SCREEN_METADATA.out.n_unobserved_plates )
}
