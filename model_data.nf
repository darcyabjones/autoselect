#!/usr/bin/env nextflow

nextflow.preview.dsl = 2

def helpMessage() {
    log.info "# autoselect"
}


process model {

    label 'python'
    label 'cpu_high'
    label 'ram_high'
    label 'time_medium'

    publishDir "${params.outdir}"

    input:
    tuple path(infile),
          val(model),
          val(response),
          val(chip),
          val(seed)
    val ntrials

    output:
    path "${infile.baseName}_${model}_${response}_${chip}_${seed}*"

    script:
    ncpu = model.startsWith("tf") ? 1 : task.cpus
    time = task.time.seconds > 3600 ? task.time.seconds - 3600 : task.time.seconds
    """
    run_model.py \
      --prefix "${infile.baseName}_${model}_${response}_${chip}_${seed}" \
      --response "${response}" \
      --chip "${chip}" \
      --stat mae \
      --model "${model}" \
      --ntrials "${ntrials}" \
      --maxtime "${time}" \
      --njobs "${ncpu}" \
      "${infile}"
    """
}


workflow {

    main:

    if ( params.help ) {
        helpMessage()
        exit 0
    }

    if ( ! params.infile ) {
        println "Nup"
        exit 1
    }

    infile = Channel.fromPath( params.infile, checkIfExists: true)

    models = Channel.of( "xgboost", "knn", "rf", "ngboost", "svr", "extratrees", "elasticnet", "elasticnet_dist", "lassolars", "lassolars_dist", "tf_basic", "tf_sum", "tf_concat", "tf_gated" )

    response = Channel.of( "resistance", "yield" )

    // 25 k and 5 k
    chips = Channel.of( 0, 9 )

    seed = Channel.of( 123 )

    combined = infile
        .combine(models)
        .combine(response)
        .combine(chips)
        .combine(seed)

    results = model(combined, 200)
}
