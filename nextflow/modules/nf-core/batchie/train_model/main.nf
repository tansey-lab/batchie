process TRAIN_MODEL {
    tag "$meta.id"
    label 'process_long'
    label 'process_high_memory'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/batchie:latest' :
        'docker.io/jeffquinnmsk/batchie:latest' }"

    input:
    tuple val(meta), path(data), val(chain_index), val(n_chains)

    output:
    tuple val(meta), path("${prefix}/thetas*.h5"), emit: thetas
    path  "versions.yml"                , emit: versions


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}"
    train_model --data ${data} \
        --chain-index ${chain_index} \
        --n-chains ${n_chains} \
        --output "${prefix}/thetas_${chain_index}.h5" \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        batchie: \$( python -c 'from importlib.metadata import version;print(version("batchie"))' )
    END_VERSIONS
    """
}
