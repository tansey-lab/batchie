process SELECT_NEXT_PLATE {
    tag "$meta.id"
    label 'process_single'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/batchie:latest' :
        'docker.io/jeffquinnmsk/batchie:latest' }"

    input:
    tuple val(meta), path(data), val(excludes), path(scores)

    output:
    tuple val(meta), env(SELECTED_PLATE)             , emit: selected_plate
    tuple val(meta), file("${prefix}/selected_plate"), emit: selected_plate_file
    path  "versions.yml"                             , emit: versions


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    def exclude_flag = excludes != null && excludes.join(" ").trim() != "" ? "--batch-plate-id ${excludes.join(" ")}" : ""
    """
    mkdir -p "${prefix}"
    select_next_plate --data ${data} \
        --scores ${scores} \
        ${exclude_flag} \
        --output "${prefix}/selected_plate" \
        ${args}

    SELECTED_PLATE=\$(cat ${prefix}/selected_plate)

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        batchie: \$( python -c 'from importlib.metadata import version;print(version("batchie"))' )
    END_VERSIONS
    """
}
