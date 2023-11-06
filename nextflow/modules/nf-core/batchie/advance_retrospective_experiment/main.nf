process ADVANCE_RETROSPECTIVE_SIMULATION {
    tag "$meta.id"
    label 'process_single'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/batchie:latest' :
        'docker.io/jeffquinnmsk/batchie:latest' }"

    input:
    tuple val(meta), path(data_masked), path(data_unmasked), path(thetas), path(batch_selection), path(experiment_tracker)

    output:
    tuple val(meta), path("${prefix}/advanced_experiment.h5"), emit: advanced_experiment
    tuple val(meta), path("${prefix}/experiment_tracker_output.json"), emit: experiment_tracker
    path  "versions.yml"                , emit: versions


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}"
    advance_retrospective_simulation \
        --thetas ${thetas} \
        --masked-screen ${data_masked} \
        --unmasked-screen ${data_unmasked} \
        --batch-selection ${batch_selection} \
        --experiment-tracker-output ${prefix}/experiment_tracker_output.json \
        --experiment-tracker-input ${experiment_tracker} \
        --screen-output ${prefix}/advanced_experiment.h5 \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        batchie: \$( python -c 'from importlib.metadata import version;print(version("batchie"))' )
    END_VERSIONS
    """
}
