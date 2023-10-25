process CALCULATE_DISTANCE_MATRIX_CHUNK {
    tag "$meta.id"
    label 'process_long'
    label 'process_high_memory'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/batchie:latest' :
        'docker.io/jeffquinnmsk/batchie:latest' }"

    input:
    tuple val(meta), path(data), path(thetas), val(chunk_index), val(n_chunks)

    output:
    tuple val(meta), path("${prefix}/distance_matrix_chunk.h5"), emit: distance_matrix_chunk
    path  "versions.yml"                , emit: versions


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir "${prefix}"
    calculate_distance_matrix --data ${data} \
        --thetas ${thetas} \
        --chunk-index ${chunk_index} \
        --n-chunks ${n_chunks} \
        --output "${prefix}/distance_matrix_chunk.h5" \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        batchie: \$( python -c 'from importlib.metadata import version;print(version("batchie"))' )
    END_VERSIONS
    """
}
