process EXTRACT_EXPERIMENT_METADATA {
    tag "$meta.id"
    label 'process_single'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/batchie:latest' :
        'docker.io/jeffquinnmsk/batchie:latest' }"

    input:
    tuple val(meta), path(data)

    output:
    tuple val(meta), path("${prefix}/experiment_metadata.json"), emit: distance_matrix_chunk
    tuple val(meta), env(n_unique_samples), emit: n_unique_samples
    tuple val(meta), env(n_unique_treatments), emit: n_unique_treatments
    tuple val(meta), env(n_observed_plates), emit: n_observed_plates
    tuple val(meta), env(n_unobserved_plates), emit: n_unobserved_plates
    tuple val(meta), env(n_plates), emit: n_plates
    tuple val(meta), env(size), emit: size
    path  "versions.yml"                , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}"
    extract_experiment_metadata --experiment ${data} \
        --output ${prefix}/experiment_metadata.json \
        ${args}

    n_unique_samples=\$(jq -r '.n_unique_samples' ${prefix}/experiment_metadata.json)
    n_unique_treatments=\$(jq -r '.n_unique_treatments' ${prefix}/experiment_metadata.json)
    n_observed_plates=\$(jq -r '.n_observed_plates' ${prefix}/experiment_metadata.json)
    n_unobserved_plates=\$(jq -r '.n_unobserved_plates' ${prefix}/experiment_metadata.json)
    n_plates=\$(jq -r '.n_plates' ${prefix}/experiment_metadata.json)
    size=\$(jq -r '.size' ${prefix}/experiment_metadata.json)

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        batchie: \$( python -c 'from importlib.metadata import version;print(version("batchie"))' )
    END_VERSIONS
    """
}
