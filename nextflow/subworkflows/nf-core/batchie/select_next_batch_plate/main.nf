include { CALCULATE_DISTANCE_MATRIX_CHUNK } from '../../../../modules/nf-core/batchie/calculate_distance_matrix_chunk/main'
include { CALCULATE_SCORE_CHUNK } from '../../../../modules/nf-core/batchie/calculate_score_chunk/main'
include { EVALUATE_MODEL } from '../../../../modules/nf-core/batchie/evaluate_model/main'
include { REVEAL_PLATE } from '../../../../modules/nf-core/batchie/reveal_plate/main'
include { SELECT_NEXT_PLATE } from '../../../../modules/nf-core/batchie/select_next_plate/main'
include { TRAIN_MODEL } from '../../../../modules/nf-core/batchie/train_model/main'


def create_parallel_sequence(meta, n_par) {
    def output = []

    for (x in (0..(n_par-1))) {
        output.add(tuple(meta, x, n_par))
    }
    return output
}


workflow SELECT_NEXT_BATCH_PLATE {
    take:
    ch_input  // channel: [ val(meta), path(screen), path(thetas), path(distance_matrix_chunks), val(n_chunks), val(reveal) ]

    main:
    ch_input.flatMap { create_parallel_sequence(it[0], it[4]) }.tap { dist_input }
    ch_input.map { tuple(it[0], it[2]) }.tap { thetas }
    ch_input.map { tuple(it[0], it[3]) }.tap { distance_matrix_chunks }

    ch_input.map { tuple(it[0], it[1]) }
        .join(thetas)
        .join(distance_matrix_chunks)
        .combine(dist_input, by: 0)
        .tap { score_chunk_input }

    CALCULATE_SCORE_CHUNK( score_chunk_input )

    ch_input.map { tuple(it[0], it[1]) }
        .join(CALCULATE_SCORE_CHUNK.out.score_chunk.groupTuple())
        .tap { select_next_plate_input }

    SELECT_NEXT_PLATE( select_next_plate_input )

    if (params.reveal) {
        ch_input.map { tuple(it[0], it[1]) }
            .join(SELECT_NEXT_PLATE.out.selected_plate.groupTuple())
            .tap { reveal_plate_input }

        REVEAL_PLATE( reveal_plate_input )
    }
}
