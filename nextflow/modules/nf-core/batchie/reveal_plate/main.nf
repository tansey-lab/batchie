process REVEAL_PLATE {
    tag "$meta.id"
    label 'process_single'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/batchie:latest' :
        'docker.io/jeffquinnmsk/batchie:latest' }"

    input:
    tuple val(meta), path(screen), val(plate_ids)

    output:
    tuple val(meta), path("${prefix}/advanced_screen.h5"), emit: advanced_screen
    path  "versions.yml"                , emit: versions


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    def plate_ids_joined = (plate_ids as Iterable).join ' '
    """
    mkdir -p "${prefix}"
    reveal_plate \
        --screen ${screen} \
        --plate-id ${plate_ids_joined} \
        --output ${prefix}/advanced_screen.h5 \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        batchie: \$( python -c 'from importlib.metadata import version;print(version("batchie"))' )
    END_VERSIONS
    """
}
