#!/usr/bin/env Rscript

VERSION = "0.0.1"

set.seed(5)

suppressPackageStartupMessages(library("optparse"))
suppressWarnings(suppressPackageStartupMessages(library("AlphaSimR")))
suppressWarnings(suppressPackageStartupMessages(library("hdf5r")))


POPULATION_SCENARIOS <- c(
  "biparental",
  "all"
)

SELECTION_SCHEMES <- c(
  "random",
  "resistance",
  "yield"
)

option_list <- list(
  make_option(
    c("-i", "--infile"),
    type="character",
    action="store",
    help="The input tsv file (required)."
  ),
  make_option(
    c("-o", "--outfile"),
    type="character",
    action="store",
    help="The output h5 file (required)."
  ),
  make_option(
    "--crossingnoise",
    type="double",
    action="store",
    default=0.0,
    help=paste(
      "Adds an amount of noise to the crossing from the current generation.",
      "In crossing, it's often the case that you'll get some random pollenation",
      "that isn't the target of your crossing scheme. This option says that a random",
      "proportion of your progeny in the next generation will come from a random",
      "pollenation of parents in the current generation."
    )
  ),
  make_option(
    "--scenario",
    type="character",
    action="store",
    default="all",
    help=paste("One of", POPULATION_SCENARIOS, collapse=", ")
  ),
  make_option(
    "--selection",
    type="character",
    action="store",
    default="random",
    help=paste("One of", SELECTION_SCHEMES, collapse=", ")
  ),
  make_option(
    "--seed",
    type="integer",
    action="store",
    default=as.integer(5),
    help="Set the random seed.",
  ),
  make_option(
    "--nprogeny",
    type="integer",
    action="store",
    default=as.integer(25),
    help=paste(
      "What should be the default number of progeny from each cross.",
      "Here we have in mind something like the smallest number of progeny you'd usually get from a cross,",
      "E.G. how many seeds you'd take from a single flower/inflorescence."
    )
  ),
  make_option(
    "--ntraininggenerations",
    type="integer",
    action="store",
    default=as.integer(3),
    help="The number of generations to run to generate the training population from the founders.",
  ),
  make_option(
    "--ngenerations",
    type="integer",
    action="store",
    default=as.integer(10),
    help="The number of generations to run for evaluation.",
  ),
  make_option(
    "--trainingsize",
    type="integer",
    action="store",
    default=as.integer(5000),
    help="The number of individuals to select for the training population.",
  ),
  make_option(
    "--ntrainingsubpops",
    type="integer",
    action="store",
    default=as.integer(1),
    help="The number of training subpopulations to be generated.",
  ),
  make_option(
    "--trainmultigenerational",
    type="logical",
    action="store_true",
    default=FALSE,
    help="Instead of using only the last generation from the training population for training, use all generations."
  ),
  make_option(
    "--backcross",
    type="logical",
    action="store_true",
    default=FALSE,
    help="Back cross each generation to F1 instead of randomly with itself."
  ),
  make_option(
    "--ncrossselect",
    type="integer",
    action="store",
    default=as.integer(100),
    help="The number of top performing progeny to select from crosses for further generations.",
  ),
  make_option(
    "--crosssize",
    type="integer",
    action="store",
    default=as.integer(1000),
    help="The desired number of crossed progeny from each generation (crosssize / nprogeny will be number of crosses).",
  ),
  make_option(
    "--nreplicates",
    type="integer",
    action="store",
    default=as.integer(8),
    help="How many replicates to use for each phenotype measurement.",
  ),
  make_option(
    "--version",
    type="logical",
    action="store_true",
    default=FALSE,
    help="Print version and exit.",
  )
)

parser <- OptionParser(
  usage = "%prog --infile in.tsv --outfile out.h5",
  option_list = option_list
)

args <- parse_args(parser)

# Make it reproducible
# Eventually this should really go in the main function, but I was a bit sloppy
# and set default arguments that should be randomised (e.g. EFFECTOR_INCIDENCE).
set.seed(args$seed)

log_stderr <- function(...) {
  cat(sprintf(...), sep='', file=stderr())
}

quit_with_err <- function(...) {
  log_stderr(...)
  quit(save = "no", status = 1, runLast = FALSE)
}

validate_file <- function(path) {
  if (is.null(path)) {
    quit_with_err("Please provide required file")
  }
}


parse_input_df <- function(path) {
  df <- read.table(
    path,
    stringsAsFactors = FALSE,
    header = TRUE,
    sep = "\t"
  )

  if(any(colnames(df)[1:3] != c("chrom", "pos", "dist"))) {
    print(colnames(df) == c("chrom", "pos", "dist"))
    quit_with_err("The first three columns of your dataframe must be ('chrom', 'pos', 'dist').")
  }

  df$pos <- as.numeric(df$pos)
  df$dist <- as.numeric(df$dist)

  df[order(df$chrom, as.numeric(df$pos)),]
}


# sets a matrix to be symmetrical, with 1s on the diagonal and replacing
# the current lower corner with the current upper.
make_symmetric <- function(m) {
  diag(m) <- 1
  m[lower.tri(m)] <- t(m)[lower.tri(m)]
  m
}


# Samples a vector of values where the sum is equal to something.
# Used to select the number of loci affecting a trait for each chromosome.
sample_with_size <- function(n, nChr, min, max) {
  test = TRUE
  while (test) {
    x <- sample(min:max, nChr, replace=TRUE)
    if(sum(x) == n) {
      test = FALSE
    }
  }
  x
}


# Samples a set number of markers for each chromosome
# proportional to the number of segreggating markers in each chromosome.
# E.g. a chrom with few markers should receive fewer markers in the snpchip.
snp_chip_density <- function(n, nLoci) {
  prop <- nLoci / sum(nLoci)
  counts <- round(n * prop)
  if (sum(counts) > n) {
    extra <- sum(counts) - n
    correction <- sample(c(rep(1, extra), rep(0, length(nLoci) - extra)), size = length(nLoci), replace = FALSE)
    counts <- counts - correction
    if (any(counts < 0)) {
      # Just in case we end up with negative numbers as result of random sampling.
      # This shouldn't happen too often.
      counts <- snp_chip_density(n, nLoci)
    }
  } else if (sum(counts) < n) {
    extra <- n - sum(counts)
    correction <- sample(c(rep(1, extra), rep(0, length(nLoci) - extra)), size = length(nLoci), replace = FALSE)
    counts <- counts + correction
  }

  if (sum(counts) != n) {
    stop(paste("The number of markers wasn't what was desired. This shouldn't happen.", sum(counts)))
  }

  return(counts)
}


prep_founder_pop <- function(df) {
  pop <- AlphaSimR::newMapPop(
    split(df$dist, df$chrom),
    lapply(split(df[, -c(1, 2, 3)], df$chrom), function(x) {t(as.matrix(x))}),
    ploidy = 2,
    inbred = TRUE
  )

  return(pop)
}


prep_sp <- function(pop) {

  yield_corA <- make_symmetric(matrix(runif(16, 0.5, 0.99), nrow = 4, ncol = 4))
  yield_corDD <- make_symmetric(matrix(runif(16, 0.2, 0.8), nrow = 4, ncol = 4)) * yield_corA
  yield_corAA <- make_symmetric(matrix(runif(16, 0.2, 0.8), nrow = 4, ncol = 4)) * yield_corA
  yield_corGxE <- make_symmetric(matrix(runif(16, 0.5, 0.95), nrow = 4, ncol = 4))

  resistance_corA <- make_symmetric(matrix(runif(64, 0.5, 0.8), nrow = 8, ncol = 8))
  resistance_corDD <- make_symmetric(matrix(runif(64, 0.4, 0.9), nrow = 8, ncol = 8)) * resistance_corA
  resistance_corAA <- make_symmetric(matrix(runif(64, 0.4, 0.9), nrow = 8, ncol = 8)) * resistance_corA
  resistance_corGxE <- make_symmetric(matrix(runif(64, 0.3, 0.8), nrow = 8, ncol = 8))

  effector_effects <- rep(10, 10)

  SP <- SimParam$new(pop)$addTraitADEG(
    nQtlPerChr=sample_with_size(10000, pop@nChr, 300, 1000),
    mean = rnorm(4, mean = 50, sd = 15),
    var =  rchisq(4, 15),
    varEnv = rchisq(4, 5),
    varGxE = rchisq(4, 3),
    meanDD = runif(4, 0.1, 0.3),
    varDD = runif(4, 0.2, 0.3),
    relAA = runif(4, 1, 3),
    corA = yield_corA,
    corDD = yield_corDD,
    corAA = yield_corAA,
    corGxE = yield_corGxE
  )$  # Yield
    addTraitADEG(
      nQtlPerChr=sample_with_size(10000, pop@nChr, 300, 1000),
      gamma = TRUE,
      shape = rnorm(4, mean = 5, sd = 1),
      mean = rnorm(4, mean = 50, sd = 10),
      var =  rchisq(4, 15),
      varEnv = rchisq(4, 5),
      varGxE = rchisq(4, 3),
      meanDD = runif(4, 0.1, 0.3),
      varDD = runif(4, 0.2, 0.3),
      relAA = runif(4, 1, 3),
      corA = yield_corA,
      corDD = yield_corDD,
      corAA = yield_corAA,
      corGxE = yield_corGxE
    )$  # YieldSkewed
    addTraitADEG(
      nQtlPerChr=sample_with_size(1000, pop@nChr, 30, 70),
      gamma = TRUE,
      shape = rnorm(8, mean = 10, sd = 2),
      mean = rnorm(8, mean = 10, sd = 3),
      var =  rchisq(8, 15),
      varEnv = rchisq(8, 5),
      varGxE = rchisq(8, 3),
      meanDD = runif(8, 0.2, 0.5),
      varDD = runif(8, 0.2, 0.5),
      relAA = runif(8, 1, 3),
      corA = resistance_corA,
      corDD = resistance_corDD,
      corAA = resistance_corAA,
      corGxE = resistance_corGxE
    )$ # Resistance
    addTraitAD(
      nQtlPerChr=sample(c(integer(pop@nChr - 1), 1),
      size = pop@nChr, replace = FALSE),
      mean = effector_effects[1],
      var = effector_effects[1] / 4, meanDD = 1
    )$ # Effector1
    addTraitAD(
      nQtlPerChr=sample(c(integer(pop@nChr - 1), 1), size = pop@nChr, replace = FALSE),
      mean = effector_effects[2],
      var = effector_effects[2] / 4,
      meanDD = 1
    )$ # Effector2
    addTraitAD(
      nQtlPerChr=sample(c(integer(pop@nChr - 1), 1), size = pop@nChr, replace = FALSE),
      mean = effector_effects[3],
      var = effector_effects[3] / 4,
      meanDD = 1
    )$ # Effector3
    addTraitAD(
      nQtlPerChr=sample(c(integer(pop@nChr - 1), 1), size = pop@nChr, replace = FALSE),
      mean = effector_effects[4],
      var = effector_effects[4] / 4,
      meanDD = 1
    )$ # Effector4
    addTraitAD(
      nQtlPerChr=sample(c(integer(pop@nChr - 1), 1),
      size = pop@nChr, replace = FALSE),
      mean = effector_effects[5],
      var = effector_effects[5] / 4,
      meanDD = 1
    )$ # Effector5
    addTraitAD(
      nQtlPerChr=sample(c(integer(pop@nChr - 1), 1), size = pop@nChr, replace = FALSE),
      mean = effector_effects[6],
      var = effector_effects[6] / 4,
      meanDD = 1
    )$ # Effector6
    addTraitAD(
      nQtlPerChr=sample(c(integer(pop@nChr - 1), 1), size = pop@nChr, replace = FALSE),
      mean = effector_effects[7],
      var = effector_effects[7] / 4,
      meanDD = 1
    )$ # Effector7
    addTraitAD(
      nQtlPerChr=sample(c(integer(pop@nChr - 1), 1), size = pop@nChr, replace = FALSE),
      mean = effector_effects[8],
      var = effector_effects[8] / 4,
      meanDD = 1
    )$ # Effector8
    addTraitAD(
      nQtlPerChr=sample(c(integer(pop@nChr - 1), 1), size = pop@nChr, replace = FALSE),
      mean = effector_effects[9],
      var = effector_effects[9] / 4,
      meanDD = 1
    )$ # Effector9
    addTraitAD(
      nQtlPerChr=sample(c(integer(pop@nChr - 1), 1), size = pop@nChr, replace = FALSE),
      mean = effector_effects[10],
      var = effector_effects[10] / 4,
      meanDD = 1
    )$ # Effector10
    addTraitA(
      nQtlPerChr=sample(c(integer(pop@nChr - 1), 1), size = pop@nChr, replace = FALSE),
      mean = 10,
      var = 5
    )$ # Monogenic
    addTraitA(
      nQtlPerChr=sample(c(integer(pop@nChr - 5), 1, 1, 1, 1, 1), size = pop@nChr, replace = FALSE),
      mean = 10,
      var = 5
    )$
    addTraitA(nQtlPerChr=sample_with_size(10, pop@nChr, 0, 2), mean = 10, var = 5)$ # 10
    addTraitA(nQtlPerChr=sample_with_size(50, pop@nChr, 1, 7), mean = 10, var = 5)$ # 50
    addTraitA(nQtlPerChr=sample_with_size(100, pop@nChr, 1, 15), mean = 10, var = 5)$ # 100
    addTraitA(nQtlPerChr=sample_with_size(500, pop@nChr, 15, 40), mean = 10, var = 5)$ # 500
    addTraitA(nQtlPerChr=sample_with_size(1000, pop@nChr, 30, 70), mean = 10, var = 5)$ # 1000
    addTraitA(nQtlPerChr=sample_with_size(10000, pop@nChr, 300, 700), mean = 10, var = 5)$ # 1000
    addTraitAD(
      nQtlPerChr=sample(c(integer(pop@nChr - 1), 1), size = pop@nChr, replace = FALSE),
      mean = 10,
      var = 5,
      meanDD = 0.2,
      varDD=0
    )$ # Monogenic
    addTraitAD(
      nQtlPerChr=sample(c(integer(pop@nChr - 5), 1, 1, 1, 1, 1), size = pop@nChr, replace = FALSE),
      mean = 10,
      var = 5,
      meanDD = 0.2,
      varDD=0
    )$
    addTraitAD(
      nQtlPerChr=sample_with_size(10, pop@nChr, 0, 2),
      mean = 10,
      var = 5,
      meanDD = 0.2,
      varDD=0.2
    )$ # 10
    addTraitAD(
      nQtlPerChr=sample_with_size(50, pop@nChr, 1, 7),
      mean = 10,
      var = 5,
      meanDD = 0.2,
      varDD = 0.2
    )$ # 50
    addTraitAD(
      nQtlPerChr=sample_with_size(100, pop@nChr, 1, 15),
      mean = 10,
      var = 5,
      meanDD = 0.2,
      varDD = 0.2
    )$ # 100
    addTraitAD(
      nQtlPerChr=sample_with_size(500, pop@nChr, 15, 40),
      mean = 10,
      var = 5,
      meanDD = 0.2,
      varDD=0.2
    )$ # 500
    addTraitAD(
      nQtlPerChr=sample_with_size(1000, pop@nChr, 30, 70),
      mean = 10,
      var = 5,
      meanDD = 0.2,
      varDD = 0.2
    )$ # 1000
    addTraitAD(
      nQtlPerChr = sample_with_size(10000, pop@nChr, 300, 700),
      mean = 10,
      var = 5,
      meanDD = 0.2,
      varDD = 0.2
    )$ # 1000
    addTraitAE(
      nQtlPerChr=sample(c(integer(pop@nChr - 2), 1, 1), size = pop@nChr, replace = FALSE),
      mean = 10,
      var = 5,
      relAA = 3
    )$ # Monogenic
    addTraitAE(
      nQtlPerChr=sample(c(integer(pop@nChr - 6), 1, 1, 1, 1, 1, 1), size = pop@nChr, replace = FALSE),
      mean = 10,
      var = 5,
      relAA = 3
    )$
    addTraitAE(nQtlPerChr=sample_with_size(10, pop@nChr, 0, 2), mean = 10, var = 5, relAA = 3)$ # 10
    addTraitAE(nQtlPerChr=sample_with_size(50, pop@nChr, 1, 7), mean = 10, var = 5, relAA = 3)$ # 50
    addTraitAE(nQtlPerChr=sample_with_size(100, pop@nChr, 1, 15), mean = 10, var = 5, relAA = 3)$ # 100
    addTraitAE(nQtlPerChr=sample_with_size(500, pop@nChr, 15, 40), mean = 10, var = 5, relAA = 3)$ # 500
    addTraitAE(nQtlPerChr=sample_with_size(1000, pop@nChr, 30, 70), mean = 10, var = 5, relAA = 3)$ # 1000
    addTraitAE(nQtlPerChr=sample_with_size(10000, pop@nChr, 300, 700), mean = 10, var = 5, relAA = 3)$ # 1000
    addTraitADE(
      nQtlPerChr=sample(c(integer(pop@nChr - 2), 1, 1), size = pop@nChr, replace = FALSE),
      mean = 10,
      var = 5,
      relAA = 3,
      meanDD = 0.2,
      varDD=0
    )$ # Monogenic
    addTraitADE(
      nQtlPerChr=sample(c(integer(pop@nChr - 6), 1, 1, 1, 1, 1, 1), size = pop@nChr, replace = FALSE),
      mean = 10,
      var = 5,
      relAA = 3,
      meanDD = 0.2,
      varDD=0
    )$
    addTraitADE(
      nQtlPerChr=sample_with_size(10, pop@nChr, 0, 2),
      mean = 10,
      var = 5,
      relAA = 3,
      meanDD = 0.2,
      varDD=0.2
    )$ # 10
    addTraitADE(
      nQtlPerChr=sample_with_size(50, pop@nChr, 1, 7),
      mean = 10,
      var = 5,
      relAA = 3,
      meanDD = 0.2,
      varDD=0.2
    )$ # 50
    addTraitADE(
      nQtlPerChr=sample_with_size(100, pop@nChr, 1, 15),
      mean = 10,
      var = 5,
      relAA = 3,
      meanDD = 0.2,
      varDD=0.2
    )$ # 100
    addTraitADE(
      nQtlPerChr=sample_with_size(500, pop@nChr, 15, 40),
      mean = 10,
      var = 5,
      relAA = 3,
      meanDD = 0.2,
      varDD=0.2
    )$ # 500
    addTraitADE(
      nQtlPerChr=sample_with_size(1000, pop@nChr, 30, 70),
      mean = 10,
      var = 5,
      relAA = 3,
      meanDD = 0.2,
      varDD=0.2
    )$ # 1000
    addTraitADE(
      nQtlPerChr=sample_with_size(10000, pop@nChr, 300, 700),
      mean = 10,
      var = 5,
      relAA = 3,
      meanDD = 0.2,
      varDD=0.2
    )$ # 1000
    setSexes("no")$
    addSnpChip(nSnpPerChr = snp_chip_density(25000, pop@nLoci))$
    addSnpChip(nSnpPerChr = snp_chip_density(25000, pop@nLoci))$
    addSnpChip(nSnpPerChr = snp_chip_density(25000, pop@nLoci))$
    addSnpChip(nSnpPerChr = snp_chip_density(25000, pop@nLoci))$
    addSnpChip(nSnpPerChr = snp_chip_density(10000, pop@nLoci))$
    addSnpChip(nSnpPerChr = snp_chip_density(10000, pop@nLoci))$
    addSnpChip(nSnpPerChr = snp_chip_density(10000, pop@nLoci))$
    addSnpChip(nSnpPerChr = snp_chip_density(10000, pop@nLoci))$
    addSnpChip(nSnpPerChr = snp_chip_density(5000, pop@nLoci))$
    addSnpChip(nSnpPerChr = snp_chip_density(5000, pop@nLoci))$
    addSnpChip(nSnpPerChr = snp_chip_density(5000, pop@nLoci))$
    addSnpChip(nSnpPerChr = snp_chip_density(5000, pop@nLoci))$
    addSnpChip(nSnpPerChr = snp_chip_density(1000, pop@nLoci))$
    addSnpChip(nSnpPerChr = snp_chip_density(1000, pop@nLoci))$
    addSnpChip(nSnpPerChr = snp_chip_density(1000, pop@nLoci))$
    addSnpChip(nSnpPerChr = snp_chip_density(1000, pop@nLoci))$
    addSnpChip(nSnpPerChr = snp_chip_density(100, pop@nLoci))$
    addSnpChip(nSnpPerChr = snp_chip_density(100, pop@nLoci))$
    addSnpChip(nSnpPerChr = snp_chip_density(100, pop@nLoci))$
    addSnpChip(nSnpPerChr = snp_chip_density(100, pop@nLoci))$
    addSnpChip(nSnpPerChr = snp_chip_density(50, pop@nLoci))$
    addSnpChip(nSnpPerChr = snp_chip_density(50, pop@nLoci))$
    addSnpChip(nSnpPerChr = snp_chip_density(50, pop@nLoci))$
    addSnpChip(nSnpPerChr = snp_chip_density(50, pop@nLoci))

  # Leaving these out because it uses sooo much ram.
  #  addSnpChip(nSnpPerChr = snp_chip_density(50000, pop@nLoci))$
  #  addSnpChip(nSnpPerChr = snp_chip_density(50000, pop@nLoci))$
  #  addSnpChip(nSnpPerChr = snp_chip_density(50000, pop@nLoci))$
  #  addSnpChip(nSnpPerChr = snp_chip_density(50000, pop@nLoci))$
  #  addSnpChip(nSnpPerChr = snp_chip_density(100000, pop@nLoci))$
  #  addSnpChip(nSnpPerChr = snp_chip_density(100000, pop@nLoci))$
  #  addSnpChip(nSnpPerChr = snp_chip_density(100000, pop@nLoci))$
  #  addSnpChip(nSnpPerChr = snp_chip_density(100000, pop@nLoci))$

  SP$setVarE(varE = c(
    rnorm(8, mean=5, sd=2),
    rnorm(8, mean=1, sd=0.25),
    rep(0.1, 10),
    rep(1, SP$nTraits - 16 - 10)
  ))
  return(SP)
}



PHENOTYPE_COLUMNS <- c(
  "yield1", "yield2", "yield3", "yield4",
  "yield_skewed1", "yield_skewed2", "yield_skewed3", "yield_skewed4",
  "resistance1", "resistance2", "resistance3", "resistance4",
  "resistance5", "resistance6", "resistance7", "resistance8",
  "effector1", "effector2", "effector3", "effector4", "effector5",
  "effector6", "effector7", "effector8", "effector9", "effector10",
  "A1", "A5", "A10", "A50", "A100", "A500", "A1000", "A10000",
  "AD1", "AD5", "AD10", "AD50", "AD100", "AD500", "AD1000", "AD10000",
  "AE2", "AE6", "AE10", "AE50", "AE100", "AE500", "AE1000", "AE10000",
  "ADE2", "ADE6", "ADE10", "ADE50", "ADE100", "ADE500", "ADE1000", "ADE10000"
)

EFFECTOR_COLUMNS <- paste0("effector", 1:10)
YIELD_COLUMNS <- paste0("yield", 1:4)
RESISTANCE_COLUMNS <- paste0("resistance", 1:8)
EFFECTOR_EFFECTS <- c(rnorm(2, 14, 2), rnorm(8, 8, 2))
EFFECTOR_FREQS <- c(0.8, 0.8, runif(8, 0.1, 0.5))

EFFECTOR_INCIDENCE <- t(apply(
  matrix(EFFECTOR_FREQS),
  MARGIN = 1,
  FUN = function(p) {
    sample(c(0, 1), 8, replace = TRUE, prob = c(1 - p, p))
  }
))

sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

get_pheno_df <- function(
  pop,
  generation,
  SP,
  incidence=EFFECTOR_INCIDENCE,
  effector_effects = EFFECTOR_EFFECTS,
  cnames = PHENOTYPE_COLUMNS,
  nreps = 8,
  effector_columns = EFFECTOR_COLUMNS,
  resistance_columns = RESISTANCE_COLUMNS
) {
  reps <- c()

  for (i in 1:nreps) {
    df <- as.data.frame(setPheno(pop, onlyPheno = TRUE, simParam = SP))
    names(df) <- cnames
    df$generation <- generation

    width <- floor(log10(nrow(df))) + 1
    df$individual <- paste0(generation, "_", formatC(1:nrow(df), width=width, flag="0"))
    df$rep <- i
    reps <- rbind(reps, df)
  }

  effector_presence_absence <- apply(
    reps[effector_columns],
    MARGIN = 2,
    FUN = function(x) {
      threshold <- min(x) + ((max(x) - min(x)) / 2)
      sigmoid(x - threshold)
    }
  )
  effector_presence_absence[effector_presence_absence < 0] <- 0

  updated_effector_effects <- t(t(effector_presence_absence) * effector_effects)

  colnames(updated_effector_effects) <- paste0("rescaled_", effector_columns)
  updated_effector_effects[updated_effector_effects < 0] <- 0

  effector_combos <- t(apply(
    updated_effector_effects,
    MARGIN = 1,
    FUN = function(x) {
      m <- incidence * x
      apply(m, MARGIN = 2, FUN = max)
    }))
  effector_combos[effector_combos < 0] <- 0

  resistance_combos <- t(apply(
    effector_presence_absence,
    MARGIN = 1,
    FUN = function(x) {
      m <- incidence * x
      apply(m, MARGIN = 2, FUN = max)
    }
  ))
  resistance_combos <- resistance_combos * reps[resistance_columns]
  resistance_combos[resistance_combos < 0] <- 0
  #random_combos <- matrix(rnorm(nrow(reps) * ncol(incidence), 0, 0.5), ncol = ncol(incidence))

  combos <- (effector_combos * 0.5) + (resistance_combos * 0.5)
  combos[combos < 0] <- 0  # Clamp

  colnames(combos) <- paste0("resistance_combo", 1:ncol(combos))

  return(cbind(
    reps[order(reps$individual, reps$rep), c(c("generation", "individual", "rep"), cnames)],
    updated_effector_effects,
    combos
  ))
}


pop_to_genotypes <- function(generation, pop, SP) {
  gt <- pullSegSiteGeno(pop, simParam = SP)
  width <- floor(log10(nrow(gt))) + 1
  rownames(gt) <- paste0(generation, "_", formatC(1:nrow(gt), width=width, flag="0"))
  colnames(gt) <- 1:ncol(gt)
  gt
}

pop_to_snpchip_genotypes <- function(generation, pop, SP, snpchip) {
  gt <- pullSnpGeno(pop, snpChip = snpchip, simParam = SP)
  width <- floor(log10(nrow(gt))) + 1
  rownames(gt) <- paste0(generation, "_", formatC(1:nrow(gt), width=width, flag="0"))
  colnames(gt) <- 1:ncol(gt)
  gt
}

fix_loci_locs <- function(pop, trait) {
  start <- c(0, cumsum(pop@nLoci[-(length(pop@nLoci))]))
  offsets <- unlist(lapply(
    1:length(trait@lociPerChr),
    FUN = function(x) {
      rep(start[x], trait@lociPerChr[x])
    }
  ))

  locs <- trait@lociLoc + offsets
  return(locs)
}

trait_to_df <- function(name, pop, trait) {
  loci_loc <- fix_loci_locs(pop, trait)
  df <- data.frame(
    trait = name,
    parameter = "intercept",
    loc1 = as.numeric(NA),
    loc2 = as.numeric(NA),
    value = trait@intercept
  )
  add <- data.frame(
    trait = name,
    parameter = "additive",
    loc1 = loci_loc,
    loc2 = as.numeric(NA),
    value = trait@addEff
  )

  if(.hasSlot(trait, "domEff")) {
    dom <- data.frame(
      trait = name,
      parameter = "dominant",
      loc1 = loci_loc,
      loc2 = as.numeric(NA),
      value = trait@domEff
    )
  } else {
    dom <- data.frame()
  }

  if(.hasSlot(trait, "epiEff")) {
    epi <- data.frame(
      trait = name,
      parameter = "epistatic",
      loc1 = loci_loc[trait@epiEff[,1]],
      loc2 = loci_loc[trait@epiEff[,2]],
      value = trait@epiEff[,3]
    )
  } else {
    epi <- data.frame()
  }

  if(.hasSlot(trait, "envVar")) {
    envvar <- data.frame(
      trait = name,
      parameter = "environmental_variance",
      loc1 = NA,
      loc2 = NA,
      value = trait@envVar
    )
  } else {
    envvar <- data.frame()
  }

  if(.hasSlot(trait, "gxeInt")) {
    envint <- data.frame(
      trait = name,
      parameter = "gxe_intercept",
      loc1 = NA,
      loc2 = NA,
      value = trait@gxeInt
    )
  } else {
    envint <- data.frame()
  }

  if(.hasSlot(trait, "gxeEff")) {
    enveff <- data.frame(
      trait = name,
      parameter = "gxe",
      loc1 = loci_loc,
      loc2 = NA,
      value = trait@gxeEff
    )
  } else {
    enveff <- data.frame()
  }

  df <- rbind(
    df,
    add,
    dom,
    epi,
    envvar,
    envint,
    enveff
  )

  return(df)
}


save_simulation <- function(
  filename,
  founders,
  pops,
  SP,
  params,
  phenotype_cnames = PHENOTYPE_COLUMNS,
  effector_effects = EFFECTOR_EFFECTS,
  effector_incidence = EFFECTOR_INCIDENCE,
  nreps = 8
) {
  h5 <- H5File$new(filename, mode = "w")
  h5[["genetic_map"]] <- unlist(founders@genMap)
  h5[["founder_genotypes"]] <- pop_to_genotypes(generation = "founders", founders, SP)
  h5[["chromosomes"]] <- data.frame(
    name = attr(founders@genMap, "dimnames")[[1]],
    start = c(1, (cumsum(founders@nLoci[-(length(founders@nLoci))]) + 1)),
    end = cumsum(founders@nLoci),
    length = founders@nLoci,
    centromere = founders@centromere
  )

  h5[["generations"]] <- unlist(lapply(names(pops), function(pname) {rep(pname, pops[[pname]]@nInd)}))
  h5[["sample_names"]] <- unlist(lapply(names(pops), function(pname) {
    width <- floor(log10(pops[[pname]]@nInd)) + 1
    paste0(pname, "_", formatC(1:pops[[pname]]@nInd, width=width, flag="0"))
  }))

  h5[["phenotypes"]] <- do.call(
    rbind,
    lapply(names(pops), function(pname) {
      get_pheno_df(pops[[pname]], pname, SP, cnames = phenotype_cnames, nreps = nreps)
    })
  )

  h5[["traits"]] <- do.call(
    rbind,
    lapply(seq_along(SP$traits), function(i) {
      trait_name <- phenotype_cnames[i]
      trait <- SP$traits[[i]]
      trait_to_df(trait_name, founders, trait)
    })
  )

  h5[["rescaled_effector_effects"]] <- effector_effects
  h5[["resistance_combo_effector_incidences"]] <- effector_incidence

  h5$create_group("snp_chips")
  sc_width <- floor(log10(length(SP$snpChips))) + 1

  for (sc_i in 1:length(SP$snpChips)) {
    h5[[paste0("snp_chips/chip_", formatC(sc_i, width=sc_width, flag="0"))]] <- do.call(
      rbind,
      lapply(
        names(pops),
        function(pop_name) {
          pop_to_snpchip_genotypes(pop_name, pops[[pop_name]], SP, sc_i)
        }
      )
    )
  }

  h5[["params"]] <- params
  h5$close_all()
}


biparental_parents <- function(
  founders,
  SP,
  npops,
  phenotype_columns = PHENOTYPE_COLUMNS,
  effector_columns = EFFECTOR_COLUMNS
) {
  founders <- newPop(founders, simParam = SP)
  df <- as.data.frame(setPheno(founders, onlyPheno = TRUE, reps = 10, simParam = SP))
  names(df) <- phenotype_columns

  dists <- as.matrix(dist(scale(df[, effector_columns])))

  xy <- t(combn(colnames(dists), 2))
  dists_df <- data.frame(x = xy[, 1], y = xy[, 2], dist = dists[xy])
  dists_df["x"] <- as.numeric(dists_df[["x"]])
  dists_df["y"] <- as.numeric(dists_df[["y"]])
  dists_df <- dists_df[order(dists_df[, "dist"], decreasing = TRUE, na.last = TRUE),]

  pops <- lapply(
    1:npops,
    function(i) {
      r <- dists_df[i,]
      F1 <- newPop(founders[c(r[["x"]], r[["y"]])], simParam = SP)
    }
  )

  names(pops) <- paste0("P", 1:npops, "F1")
  return(pops)
}


generate_training_pop <- function(
  founders,
  F1,
  SP,
  subpop,
  ngenerations,
  size,
  default_nprogeny = 25,
  randomcrosses = 0
) {
  if (randomcrosses < 0) {
    randomcrosses <- 0
  } else if (randomcrosses > 1) {
    randomcrosses <- 1
  }

  pops <- list()

  if((ngenerations == 0) & (size > nInd(F1))) {
    quit_with_err("Cannot create the specified training population size with this number of generations for this scheme.")
  } else if(ngenerations == 0) {
    pops[[paste0("P", subpop, "F1")]] <- F1[sample(1:nInd(F1), size = size, replace = FALSE)]
    return(pops)
  } else {
    pops[[paste0("P", subpop, "F1")]] <- F1
  }

  FPREV <- c(founders, F1)
  FPARENTS <- F1

  for(i in 2:(ngenerations + 1)){
    nprogeny <- default_nprogeny
    ncrosses <- ceiling(size / nprogeny)

    if (ncrosses > floor(nInd(FPARENTS) * (nInd(FPARENTS) - 1) / 2)) {
      ncrosses <- floor(nInd(FPARENTS) * (nInd(FPARENTS) - 1) / 2)
      nprogeny <- ceiling(size / ncrosses)
    }

    if (randomcrosses == 0) {
      FN <- randCross(FPARENTS, nCrosses = ncrosses, nProgeny = nprogeny, simParam = SP)
      FN <- FN[sample(1:nInd(FN), size = size, replace = FALSE)]
    } else if (randomcrosses == 1) {
      FN <- randCross2(FPARENTS, FPREV, nCrosses = ncrosses, nProgeny = nprogeny, simParam = SP)
      FN <- FN[sample(1:nInd(FN), size = size, replace = FALSE)]
    } else {
      n <- round(size * randomcrosses)
      FN <- randCross(FPARENTS, nCrosses = ncrosses, nProgeny = nprogeny, simParam = SP)
      FN <- FN[sample(1:nInd(FN), size = n, replace = FALSE)]

      FN_random <- randCross2(
        FPARENTS,
        FPREV,
        nCrosses = ncrosses,
        nProgeny = nprogeny,
        simParam = SP
      )
      FN_random <- FN_random[sample(1:nInd(FN_random), size = (size - n), replace = FALSE)]
      FN <- c(FN, FN_random)
    }

    pops[[paste0("P", subpop, "F", i)]] <- FN
    FPREV <- c(FPARENTS, FN)
    FPARENTS <- FN
  }

  return(pops)
}


run_crosses <- function(
  previous,
  training,
  SP,
  start_generation,
  ngenerations,
  nselect,
  size,
  selection_scheme,
  default_nprogeny,
  randomcrosses,
  backcross_pop = NULL,
  yield_columns = YIELD_COLUMNS,
  resistance_columns = RESISTANCE_COLUMNS,
  phenotype_columns = PHENOTYPE_COLUMNS
) {
  pops <- list()

  if (selection_scheme == "random") {
    select_use = "rand"
  } else {
    select_use = "gv"
  }

  if (selection_scheme == "random") {
    select_trait = 1
  } else if (selection_scheme == "resistance") {
    select_trait = function(d) {
      rowMeans(d[, which(phenotype_columns %in% resistance_columns)])
    }
  } else if (selection_scheme == "yield") {
    select_trait = function(d) {
      rowMeans(d[, which(phenotype_columns %in% yield_columns)])
    }
  } else {
    quit_with_err("Invalid selection scheme.")
  }

  FPARENTS = training

  if (!is.null(backcross_pop)) {
    FPREV <- c(training, previous, backcross_pop)
  } else {
    FPREV <- c(training, previous)
  }

  for (generation in start_generation:(start_generation + ngenerations)) {
    nprogeny <- default_nprogeny
    ncrosses <- ceiling(size / nprogeny)

    if (ncrosses > floor(nselect * (nselect - 1) / 2)) {
      ncrosses <- floor(nselect * (nselect - 1) / 2)
      nprogeny <- ceiling(size / ncrosses)
    }

    top_performers <- selectInd(
      FPARENTS,
      nInd = nselect,
      trait = select_trait,
      use = select_use,
      selectTop = (selection_scheme != "resistance"),
      simParam = SP
    )

    if (!is.null(backcross_pop)) {
      cross_pop <- backcross_pop
    } else {
      cross_pop <- top_performers
    }

    if (randomcrosses == 0) {
      FN <- randCross2(
        top_performers,
        cross_pop,
        nCrosses = ncrosses,
        nProgeny = nprogeny,
        simParam = SP
      )
      FN <- FN[sample(1:nInd(FN), size = size, replace = FALSE)]
    } else if (randomcrosses == 1) {
      FN <- randCross2(
        top_performers,
        FPREV,
        nCrosses = ncrosses,
        nProgeny = nprogeny,
        simParam = SP
      )
      FN <- FN[sample(1:nInd(FN), size = size, replace = FALSE)]
    } else {
      n <- round(size * randomcrosses)
      FN <- randCross2(
        top_performers,
        cross_pop,
        nCrosses = ncrosses,
        nProgeny = nprogeny,
        simParam = SP
      )
      FN <- FN[sample(1:nInd(FN), size = n, replace = FALSE)]

      FN_random <- randCross2(
        top_performers,
        FPREV,
        nCrosses = ncrosses,
        nProgeny = nprogeny,
        simParam = SP
      )
      FN_random <- FN_random[sample(1:nInd(FN_random), size = (size - n), replace = FALSE)]
      FN <- c(FN, FN_random)
    }

    pops[[paste0("F", generation)]] <- FN

    if (!is.null(backcross_pop)) {
      FPREV <- c(FPARENTS, FN, backcross_pop)
    } else {
      FPREV <- c(FPARENTS, FN)
    }
    FPARENTS <- FN
  }

  return(pops)
}


main <- function(args) {
  if (args$version) {
    cat(VERSION, file=stdout())
    quit(save = "no", status = 0, runLast = FALSE)
  }

  validate_file(args$infile)
  validate_file(args$outfile)

  if (!(args$scenario %in% POPULATION_SCENARIOS)) {
    quit_with_err(paste("scenario must be one of", POPULATION_SCENARIOS, collapse=", "))
  }

  if (!(args$selection %in% SELECTION_SCHEMES)) {
    quit_with_err(paste("selection must be one of", SELECTION_SCHEMES, collapse=", "))
  }

  df <- parse_input_df(args$infile)
  founders <- prep_founder_pop(df)

  SP <- prep_sp(founders)

  if (args$scenario == "biparental") {
    F1 <- biparental_parents(
      founders,
      SP,
      args$ntrainingsubpops,
      phenotype_columns = PHENOTYPE_COLUMNS,
      effector_columns = EFFECTOR_COLUMNS
    )
  } else if (args$scenario == "all") {
    F1 <- rep(list(newPop(founders, simParam = SP)), args$ntrainingsubpops)
    names(F1) <- paste0("P", 1:args$ntrainingsubpops, "F1")
  }

  pops <- unlist(lapply(
    1:args$ntrainingsubpops,
    function(i) {
      generate_training_pop(
        newPop(founders, simParam = SP),
        F1[[paste0("P", i, "F1")]],
        SP,
        subpop = i,
        ngenerations = args$ntraininggenerations,
        size = args$trainingsize,
        default_nprogeny = args$nprogeny,
        randomcrosses = args$crossingnoise
      )
    }
  ))

  # Because in the all scenario all of the F1s will be the same,
  # we remove the F1s that are duplicated for each pop and
  # replace it with a single one.
  if (args$scenario == "all") {
    pops <- pops[!endsWith(names(pops), "F1")]
    pops[["F1"]] = newPop(founders, simParam = SP)
  }

  if (args$trainmultigenerational) {
    training <- pops
    names(training) <- NULL
  } else {
    training <- lapply(
      1:args$ntrainingsubpops,
      function(i) { pops[[paste0("P", i, "F", args$ntraininggenerations)]] }
    )
    names(training) <- NULL
  }

  # This groups all pops into one.
  training <- do.call("c", training)
  training <- training[sample(1:nInd(training), size = args$trainingsize, replace = FALSE)]
  pops[["training"]] <- training

  if (args$trainmultigenerational) {
    previous_pop <- training
  } else if (args$ntraininggenerations <= 1) {
    previous_pop <- newPop(founders, simParam = SP)
  } else {
    previous_pop <- pops[!endsWith(names(pops), paste0("F", args$ntraininggenerations - 1))]
    names(previous_pop) <- NULL
    previous_pop <- do.call("c", previous_pop)
  }

  if (args$backcross) {
    backcross_pop <- pops[!endsWith(names(pops), "F1")]
    names(backcross_pop) <- NULL
    backcross_pop <- do.call("c", backcross_pop)
  } else {
    backcross_pop <- NULL
  }

  crosses <- run_crosses(
    previous_pop,
    training,
    SP,
    start_generation = args$ntraininggenerations + 1,
    ngenerations = args$ngenerations,
    nselect = args$ncrossselect,
    size = args$crosssize,
    selection_scheme = args$selection,
    default_nprogeny = args$nprogeny,
    randomcrosses = args$crossingnoise,
    backcross_pop = backcross_pop,
    yield_columns = YIELD_COLUMNS,
    resistance_columns = RESISTANCE_COLUMNS,
    phenotype_columns = PHENOTYPE_COLUMNS
  )

  params <- args[!(names(args) %in% c("infile", "outfile", "version", "help"))]
  paramsul <- unlist(params)
  params <- data.frame(param = names(paramsul), value = as.character(paramsul))

  save_simulation(
    args$outfile,
    founders,
    c(pops, crosses),
    SP,
    phenotype_cnames = PHENOTYPE_COLUMNS,
    effector_effects = EFFECTOR_EFFECTS,
    effector_incidence = EFFECTOR_INCIDENCE,
    nreps = args$nreplicates,
    params = params
  )
}


main(args)
