include { CALCULATE_DISTANCE_MATRIX_CHUNK } from '../../../../modules/nf-core/batchie/calculate_distance_matrix_chunk/main'
include { SELECT_NEXT_BATCH } from '../../../../modules/nf-core/batchie/select_next_batch/main'
include { TRAIN_MODEL } from '../../../../modules/nf-core/batchie/train_model/main'


def create_parallel_sequence(meta, n_par) {
    def output = []

    for (x in (0..(n_par-1))) {
        output.add(tuple(meta, x, n_par))
    }
    return output
}


workflow RUN_PROSPECTIVE_STEP {
    take:
    ch_input  // channel: [ val(meta), path(experiment), val(n_chains), val(n_chunks) ]

    main:
    ch_input.flatMap { create_parallel_sequence(it[0], it[2]) }.tap { chain_sequence }

    ch_input.map { tuple(it[0], it[1]) }.combine(chain_sequence, by: 0).tap { train_model_input }

    TRAIN_MODEL( train_model_input )

    ch_input.flatMap { create_parallel_sequence(it[0], it[3]) }.tap { dist_input }

    ch_input
        .map { tuple(it[0], it[1]) }
        .join(TRAIN_MODEL.out.thetas.groupTuple())
        .combine(dist_input, by: 0).tap { meta_exp_theta_chunk_idx_n_chunks }

    CALCULATE_DISTANCE_MATRIX_CHUNK( meta_exp_theta_chunk_idx_n_chunks )

    meta_exp_theta_dist = ch_input.map { tuple(it[0], it[1]) }
        .join(TRAIN_MODEL.out.thetas.groupTuple())
        .join(CALCULATE_DISTANCE_MATRIX_CHUNK.out.distance_matrix_chunk.groupTuple())
        .tap { meta_exp_theta_dist }

    SELECT_NEXT_BATCH( meta_exp_theta_dist )

    emit:
    ch_output       = SELECT_NEXT_BATCH.out.selected_plates
}
