process CALCULATE_SCORE_CHUNK {
    tag "$meta.id"
    label 'process_long'
    label 'process_high_memory'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/batchie:latest' :
        'docker.io/jeffquinnmsk/batchie:latest' }"

    input:
    tuple val(meta), path(data), path(thetas), path(distance_matrix), val(excludes), val(chunk_index), val(n_chunks)

    output:
    tuple val(meta), path("${prefix}/score_chunk*.h5"), emit: score_chunk
    path  "versions.yml"                , emit: versions


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    def exclude_flag = excludes != null && excludes.join(" ").trim() != "" ? "--batch-plate-ids ${excludes.join(" ")}" : ""
    """
    mkdir -p "${prefix}"
    calculate_scores --data ${data} \
        --thetas ${thetas} \
        --distance-matrix ${distance_matrix} \
        --chunk-index ${chunk_index} \
        --n-chunks ${n_chunks} \
        ${exclude_flag} \
        --output "${prefix}/score_chunk_${chunk_index}.h5" \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        batchie: \$( python -c 'from importlib.metadata import version;print(version("batchie"))' )
    END_VERSIONS
    """
}
