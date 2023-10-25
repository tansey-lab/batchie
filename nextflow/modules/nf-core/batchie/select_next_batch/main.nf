process SELECT_NEXT_BATCH {
    tag "$meta.id"
    label 'process_long'
    label 'process_high_memory'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/batchie:latest' :
        'docker.io/jeffquinnmsk/batchie:latest' }"

    input:
    tuple val(meta), path(data), path(thetas), path(distance_matrix)

    output:
    tuple val(meta), path("${prefix}/selected_plates.json"), emit: selected_plates
    path  "versions.yml"                , emit: versions


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}"
    select_next_batch --data ${data} \
        --thetas ${thetas} \
        --distance-matrix ${distance_matrix} \
        --output "${prefix}/selected_plates.json" \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        batchie: \$( python -c 'from importlib.metadata import version;print(version("batchie"))' )
    END_VERSIONS
    """
}
