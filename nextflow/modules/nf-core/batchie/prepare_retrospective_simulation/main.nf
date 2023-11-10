process PREPARE_RETROSPECTIVE_SIMULATION {
    tag "$meta.id"
    label 'process_single'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/batchie:latest' :
        'docker.io/jeffquinnmsk/batchie:latest' }"

    input:
    tuple val(meta), path(data)

    output:
    tuple val(meta), path("${prefix}/training.screen.h5"), emit: training_screen
    tuple val(meta), path("${prefix}/test.screen.h5"), emit: test_screen
    path  "versions.yml"                , emit: versions


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}"
    prepare_retrospective_simulation \
        --data ${data} \
        --training-output ${prefix}/training.screen.h5 \
        --test-output ${prefix}/test.screen.h5 \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        batchie: \$( python -c 'from importlib.metadata import version;print(version("batchie"))' )
    END_VERSIONS
    """
}
