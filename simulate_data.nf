#!/usr/bin/env nextflow

nextflow.preview.dsl = 2

def helpMessage() {
    log.info "# autoselect"
}


process gen {

    label 'R'
    label 'cpu_low'
    label 'ram_high'
    label 'time_medium'

    publishDir "${params.outdir}"

    input:
    tuple path("in.tsv"),
          val(crossingnoise),
          val(scenario),
          val(selection),
          val(seed),
          val(nprogeny),
          val(ntraininggenerations),
          val(ngenerations),
          val(trainingsize),
          val(ntrainingsubpops),
          val(trainmultigenerational),
          val(backcross),
          val(ncrossselect),
          val(crosssize),
          val(nreplicates)

    output:
    path "${scenario}_${selection}_${crossingnoise}_${seed}_${nprogeny}_${ntraininggenerations}_${ngenerations}_${trainingsize}_${ntrainingsubpops}_${trainmultigenerational}_${backcross}_${ncrossselect}_${crosssize}_${nreplicates}.h5"

    script:
    assert ["biparental", "all"].contains(scenario)
    assert ["random", "resistance", "yield"].contains(selection)
    tmg = trainmultigenerational ? "--trainmultigenerational" : " "
    bcr = backcross ? "--backcross" : " "

    """
    generate_dataset.R \
      --infile "in.tsv" \
      --outfile "${scenario}_${selection}_${crossingnoise}_${seed}_${nprogeny}_${ntraininggenerations}_${ngenerations}_${trainingsize}_${ntrainingsubpops}_${trainmultigenerational}_${backcross}_${ncrossselect}_${crosssize}_${nreplicates}.h5" \
      --crossingnoise "${crossingnoise}" \
      --scenario "${scenario}" \
      --selection "${selection}" \
      --seed "${seed}" \
      --nprogeny "${nprogeny}" \
      --ntraininggenerations "${ntraininggenerations}" \
      --ngenerations "${ngenerations}" \
      --trainingsize "${trainingsize}" \
      --ntrainingsubpops "${ntrainingsubpops}" \
      ${tmg} \
      ${bcr} \
      --ncrossselect "${ncrossselect}" \
      --crosssize "${crosssize}" \
      --nreplicates "${nreplicates}" 
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
    crossingnoise = Channel.of( 0, 0.5 )
    scenario = Channel.of( "biparental", "all" )
    selection = Channel.value( "random" ) 
    seed = Channel.value( 5 ) // Channel.of( 3, 7, 123, 256 )
    nprogeny = Channel.value( 25 )
    trainmultigenerational = Channel.of( true, false )
    ntraininggenerations = Channel.of( 3, 6 )
    ngenerations = Channel.of( 10 )
    trainingsize = Channel.of( 200, 5000 )
    ntrainingsubpops = Channel.of( 1, 8 )
    backcross = Channel.value( false )
    ncrossselect = Channel.value( 100 )
    crosssize = Channel.value( 1000 )
    nreplicates = Channel.value( 6 )

    combined = infile
        .combine(crossingnoise)
        .combine(scenario)
        .combine(selection)
        .combine(seed)
        .combine(nprogeny)
        .combine(ntraininggenerations)
        .combine(ngenerations)
        .combine(trainingsize)
        .combine(ntrainingsubpops)
        .combine(trainmultigenerational)
        .combine(backcross)
        .combine(ncrossselect)
        .combine(crosssize)
        .combine(nreplicates)

   h5s = gen(combined) 
}
