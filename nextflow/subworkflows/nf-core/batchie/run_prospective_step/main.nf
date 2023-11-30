include { CALCULATE_DISTANCE_MATRIX_CHUNK } from '../../../../modules/nf-core/batchie/calculate_distance_matrix_chunk/main'
include { CALCULATE_SCORE_CHUNK } from '../../../../modules/nf-core/batchie/calculate_score_chunk/main'
include { SELECT_NEXT_PLATE } from '../../../../modules/nf-core/batchie/select_next_plate/main'
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

    ch_input.map { tuple(it[0], []) }.tap { excludes }

    ch_input
        .map { tuple(it[0], it[1]) }
        .join(TRAIN_MODEL.out.thetas.groupTuple())
        .combine(dist_input, by: 0).tap { calculate_distance_matrix_chunk_input }

    CALCULATE_DISTANCE_MATRIX_CHUNK( calculate_distance_matrix_chunk_input )

    ch_input.map { tuple(it[0], it[1]) }
        .join(TRAIN_MODEL.out.thetas.groupTuple())
        .join(CALCULATE_DISTANCE_MATRIX_CHUNK.out.distance_matrix_chunk.groupTuple())
        .join(excludes)
        .combine(dist_input, by: 0)
        .tap { score_chunk_input }

    CALCULATE_SCORE_CHUNK( score_chunk_input )

    ch_input.map { tuple(it[0], it[1]) }
        .join(excludes)
        .join(CALCULATE_SCORE_CHUNK.out.score_chunk.groupTuple())
        .tap { select_next_plate_input }

    SELECT_NEXT_PLATE( select_next_plate_input )

    emit:
    ch_output       = SELECT_NEXT_PLATE.out.selected_plate
}
