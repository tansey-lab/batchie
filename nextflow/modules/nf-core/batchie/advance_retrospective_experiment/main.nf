process ADVANCE_RETOSPECTIVE_EXPERIMENT {
    tag "$meta.id"
    label 'process_single'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/batchie:latest' :
        'docker.io/jeffquinnmsk/batchie:latest' }"

    input:
    tuple val(meta), path(data), path(thetas), path(plate_selection), path(experiment_tracker),

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
    mkdir "${prefix}"
    advance_retrospective_experiment --data ${data} \


        parser.add_argument("--unmasked-experiment", type=str, required=True)
    parser.add_argument("--masked-experiment", type=str, required=True)
    parser.add_argument("--batch-selection", type=str, required=True)
    parser.add_argument("--experiment-output", type=str, required=True)
    parser.add_argument("--experiment-tracker-input", type=str, required=True)
    parser.add_argument("--experiment-tracker-output", type=str, required=True)
    parser.add_argument("--samples", type=str, required=True, nargs="+")
    parser.add_argument("--model", type=str, required=True)


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
