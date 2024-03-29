include { CALCULATE_SCORE_CHUNK } from '../../../../modules/nf-core/batchie/calculate_score_chunk/main'
include { REVEAL_PLATE } from '../../../../modules/nf-core/batchie/reveal_plate/main'
include { SELECT_NEXT_PLATE } from '../../../../modules/nf-core/batchie/select_next_plate/main'


def create_parallel_sequence(meta, n_par) {
    def output = []

    for (x in (0..(n_par-1))) {
        output.add(tuple(meta, x, n_par))
    }
    return output
}


workflow SELECT_NEXT_BATCH_PLATE {
    take:
    ch_input  // channel: [ val(meta), path(screen), path(thetas), path(distance_matrix_chunks), val(n_chunks), val(reveal), val(excludes) ]

    main:
    ch_input.flatMap { create_parallel_sequence(it[0], it[4]) }.tap { dist_input }
    ch_input.map { tuple(it[0], it[2]) }.tap { thetas }
    ch_input.map { tuple(it[0], it[3]) }.tap { distance_matrix_chunks }
    ch_input.map { tuple(it[0], it[6]) }.tap { excludes }

    ch_input.map { tuple(it[0], it[1], it[2], it[3], it[6]) }
        .combine(dist_input, by: 0)
        .tap { score_chunk_input }

    CALCULATE_SCORE_CHUNK( score_chunk_input )

    ch_input.map { tuple(it[0], it[1], it[6]) }
        .join(CALCULATE_SCORE_CHUNK.out.score_chunk.groupTuple())
        .tap { select_next_plate_input }

    SELECT_NEXT_PLATE( select_next_plate_input )

    output = ch_input.map { tuple(it[0], it[1]) }

    if (params.reveal) {
        ch_input.map { tuple(it[0], it[1]) }
            .join(SELECT_NEXT_PLATE.out.selected_plate.groupTuple())
            .tap { reveal_plate_input }

        REVEAL_PLATE( reveal_plate_input )

        output = REVEAL_PLATE.out.advanced_screen
    }

    emit:
    ch_output       = output
}
