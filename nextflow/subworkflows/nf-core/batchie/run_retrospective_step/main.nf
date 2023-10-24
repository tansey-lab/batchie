include { CALCULATE_DISTANCE_METRIC_CHUNK } from '../../../modules/nf-core/batchie/calculate_distance_metric_chunk/main'
include { EXTRACT_EXPERIMENT_METADATA } from '../../../modules/nf-core/batchie/extract_experiment_metadata/main'
include { SELECT_NEXT_BATCH } from '../../../modules/nf-core/batchie/select_next_batch/main'
include { TRAIN_MODEL } from '../../../modules/nf-core/batchie/train_model/main'


def create_parallel_sequence(meta, n_chains) {
    output = []

    for (x in (0..(n_chains-1))) {
        output.add(tuple(meta, x, n_chains))
    }
    return output
}



workflow RUN_RETROSPECTIVE_STEP {

    take:
    ch_input  // channel: [ val(meta), path(masked_experiment), path(unmasked_experiment), path(experiment_tracker), val(n_chains), val(n_chunks) ]

    main:

    EXTRACT_EXPERIMENT_METADATA( ch_input.map { tuple(it[0], it[1]) } )

    train_model_input = ch_input.map { it[0], it[1] }.cross(ch_input.flatMap { create_parallel_sequence(it[0], it[4]) } ))

    TRAIN_MODEL( train_model_input )

    dist_input = ch_input.flatMap { create_parallel_sequence(it[0], it[5]) }

    meta_exp_theta_chunk_idx_n_chunks = ch_input.map { tuple(it[0], it[1]) }.join(TRAIN_MODEL.thetas.groupTuple()).cross(dist_input)

    CALCULATE_DISTANCE_METRIC_CHUNK( meta_exp_theta_chunk_idx_n_chunks )

    meta_exp_theta_dist = ch_input.map { tuple(it[0], it[1]) }
        .join(TRAIN_MODEL.thetas.groupTuple())
        .join(CALCULATE_DISTANCE_METRIC_CHUNK.distance_matrix_chunk.groupTuple())

    SELECT_NEXT_BATCH( meta_exp_theta_dist )




    emit:
    cv_lambda       = BAYESTME_READ_PHENOTYPE_SELECTION_RESULTS.out.lambda
    cv_n_cell_types = BAYESTME_READ_PHENOTYPE_SELECTION_RESULTS.out.n_components
    cv_plots        = BAYESTME_READ_PHENOTYPE_SELECTION_RESULTS.out.plots
    versions        = BAYESTME_READ_PHENOTYPE_SELECTION_RESULTS.out.versions
}
