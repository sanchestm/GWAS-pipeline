#!/usr/bin/env Rscript

# GRM-based rrBLUP and Bayesian GBLUP benchmark.
#
# Input traits CSV:
#   first column: animal ID (read as character and preserved exactly)
#   remaining columns: numeric traits; NA is allowed
#
# Output CSV:
#   same IDs, row order, and trait names as the input

usage <- function(status = 0L) {
  cat(
"Usage:
  Rscript rGblup.R --model rrblup --grms GRM --re 0 [options] TRAITS.csv OUT.csv
  Rscript rGblup.R --model bglr --grms GRM [GRM2 ...] --re 0|1|UN|DIAG [options] TRAITS.csv OUT.csv

Required:
  --model, -m        rrblup or bglr
  --grms, -g         one or more GCTA GRM prefixes; put --re after the last prefix
  --re, -r           0: separate univariate fits
                     1 or UN: one multitrait fit with unstructured residual covariance
                     DIAG: one multitrait fit with diagonal residual covariance

Prediction target:
  --prediction       genetic_mean (default), breeding_value, or phenotype
  --blup, -b          backward-compatible alias for --prediction genetic_mean
  --phenotype, -p     alias for --prediction phenotype

  genetic_mean       trait intercept + genomic breeding value
  breeding_value     genomic random effect only (no trait intercept)
  phenotype          genetic_mean plus the conditional residual mean for missing
                     cells; requires bglr with --re 1 or --re UN

BGLR options:
  --nIter             total Gibbs iterations; default 6000
  --burnIn            burn-in iterations; default 1500
  --thin              thinning interval; default 1
  --seed              random seed; default 1
  --R2                prior proportion of covariance explained; default 0.5

Other options:
  --jitter            value added to each GRM diagonal; default 1e-8
  --verbose           print sampler progress
  --help, -h          show this help

Examples:
  Rscript rGblup.R -m rrblup -g data/all --re 0 train.csv rrblup.csv
  Rscript rGblup.R -m bglr -g data/all --re 0 --nIter 6000 --burnIn 1500 train.csv bglr_uni.csv
  Rscript rGblup.R -m bglr -g data/all --re UN --prediction phenotype --nIter 6000 --burnIn 1500 train.csv bglr_multi.csv
"
  )
  quit(status = status)
}

need_value <- function(args, i, flag) {
  if (i >= length(args)) stop("Missing value after ", flag, call. = FALSE)
  args[i + 1L]
}

parse_args <- function(args) {
  opt <- list(model = NULL, grms = character(), re = NULL,
              prediction = "genetic_mean", nIter = 6000L, burnIn = 1500L,
              thin = 1L, seed = 1L, R2 = 0.5, jitter = 1e-8,
              verbose = FALSE)
  positional <- character()
  known_flags <- c("--model", "-m", "--grms", "-g", "--re", "-r",
                   "--prediction", "--blup", "-b", "--phenotype", "-p",
                   "--nIter", "--niter", "--burnIn", "--burnin", "--thin",
                   "--seed", "--R2", "--r2", "--jitter", "--verbose",
                   "--help", "-h")
  i <- 1L
  while (i <= length(args)) {
    tok <- args[i]
    if (tok %in% c("--help", "-h")) usage(0L)
    if (tok %in% c("--model", "-m")) {
      opt$model <- tolower(need_value(args, i, tok)); i <- i + 2L; next
    }
    if (tok %in% c("--re", "-r")) {
      opt$re <- toupper(need_value(args, i, tok)); i <- i + 2L; next
    }
    if (tok == "--prediction") {
      opt$prediction <- tolower(need_value(args, i, tok)); i <- i + 2L; next
    }
    if (tok %in% c("--blup", "-b")) {
      opt$prediction <- "genetic_mean"; i <- i + 1L; next
    }
    if (tok %in% c("--phenotype", "-p")) {
      opt$prediction <- "phenotype"; i <- i + 1L; next
    }
    if (tok %in% c("--nIter", "--niter")) {
      opt$nIter <- as.integer(need_value(args, i, tok)); i <- i + 2L; next
    }
    if (tok %in% c("--burnIn", "--burnin")) {
      opt$burnIn <- as.integer(need_value(args, i, tok)); i <- i + 2L; next
    }
    if (tok == "--thin") {
      opt$thin <- as.integer(need_value(args, i, tok)); i <- i + 2L; next
    }
    if (tok == "--seed") {
      opt$seed <- as.integer(need_value(args, i, tok)); i <- i + 2L; next
    }
    if (tok %in% c("--R2", "--r2")) {
      opt$R2 <- as.numeric(need_value(args, i, tok)); i <- i + 2L; next
    }
    if (tok == "--jitter") {
      opt$jitter <- as.numeric(need_value(args, i, tok)); i <- i + 2L; next
    }
    if (tok == "--verbose") {
      opt$verbose <- TRUE; i <- i + 1L; next
    }
    if (tok %in% c("--grms", "-g")) {
      i <- i + 1L
      start <- i
      while (i <= length(args) && !(args[i] %in% known_flags)) {
        opt$grms <- c(opt$grms, args[i]); i <- i + 1L
      }
      if (i == start) stop("No GRM prefix after ", tok, call. = FALSE)
      next
    }
    if (startsWith(tok, "-")) stop("Unknown option: ", tok, call. = FALSE)
    positional <- c(positional, tok)
    i <- i + 1L
  }
  list(opt = opt, positional = positional)
}

validate_options <- function(opt, positional) {
  if (is.null(opt$model) || !opt$model %in% c("rrblup", "bglr"))
    stop("--model must be rrblup or bglr", call. = FALSE)
  if (!length(opt$grms)) stop("At least one GRM is required", call. = FALSE)
  if (is.null(opt$re) || !opt$re %in% c("0", "1", "UN", "DIAG"))
    stop("--re must be 0, 1, UN, or DIAG", call. = FALSE)
  if (!opt$prediction %in% c("genetic_mean", "breeding_value", "phenotype"))
    stop("--prediction must be genetic_mean, breeding_value, or phenotype", call. = FALSE)
  if (length(positional) != 2L)
    stop("Expected TRAITS.csv and OUT.csv after the options", call. = FALSE)
  if (opt$model == "rrblup" && length(opt$grms) != 1L)
    stop("rrBLUP supports exactly one GRM", call. = FALSE)
  if (opt$model == "rrblup" && opt$re != "0")
    stop("rrBLUP requires --re 0", call. = FALSE)
  if (opt$prediction == "phenotype" && (opt$model != "bglr" || !opt$re %in% c("1", "UN")))
    stop("Conditional phenotype prediction requires --model bglr --re 1|UN", call. = FALSE)
  if (anyNA(c(opt$nIter, opt$burnIn, opt$thin, opt$seed, opt$R2, opt$jitter)))
    stop("A numeric option could not be parsed", call. = FALSE)
  if (opt$nIter <= opt$burnIn || opt$burnIn < 0L || opt$thin < 1L)
    stop("Require nIter > burnIn >= 0 and thin >= 1", call. = FALSE)
  if (opt$R2 <= 0 || opt$R2 >= 1) stop("--R2 must lie strictly between 0 and 1", call. = FALSE)
  if (opt$jitter < 0) stop("--jitter must be nonnegative", call. = FALSE)
}

norm_grm_prefix <- function(x) {
  x <- sub("\\.grm\\.N\\.bin$", "", x, ignore.case = TRUE)
  x <- sub("\\.grm\\.bin$", "", x, ignore.case = TRUE)
  x <- sub("\\.grm\\.id$", "", x, ignore.case = TRUE)
  sub("\\.grm$", "", x, ignore.case = TRUE)
}

clean_ids <- function(x) trimws(as.character(x))

read_traits_csv <- function(path) {
  if (!file.exists(path)) stop("Missing traits file: ", path, call. = FALSE)
  raw <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE,
                  colClasses = "character", na.strings = c("", "NA", "NaN"))
  if (ncol(raw) < 2L) stop("Trait CSV needs an ID column and at least one trait", call. = FALSE)
  id_name <- names(raw)[1L]
  ids <- clean_ids(raw[[1L]])
  if (anyNA(ids) || any(ids == "")) stop("Animal IDs cannot be missing or empty", call. = FALSE)
  if (anyDuplicated(ids)) stop("Duplicate animal ID: ", ids[duplicated(ids)][1L], call. = FALSE)
  traits <- raw[-1L]
  for (j in seq_along(traits)) {
    old <- traits[[j]]
    new <- suppressWarnings(as.numeric(old))
    bad <- !is.na(old) & is.na(new)
    if (any(bad)) stop("Non-numeric value in trait '", names(traits)[j],
                       "': ", old[which(bad)[1L]], call. = FALSE)
    traits[[j]] <- new
  }
  rownames(traits) <- ids
  list(id_name = id_name, ids = ids, traits = traits)
}

align_grm <- function(K, fam, trait_ids, label) {
  if (!is.matrix(K) || nrow(K) != ncol(K)) stop("GRM is not square: ", label, call. = FALSE)
  if (any(!is.finite(K))) stop("GRM contains non-finite values: ", label, call. = FALSE)
  candidates <- list()
  if (!is.null(rownames(K))) candidates[["matrix row names"]] <- clean_ids(rownames(K))
  if (!is.null(fam)) {
    if (nrow(fam) != nrow(K)) stop("GRM fam table has the wrong number of rows: ", label, call. = FALSE)
    fam_col <- if ("fam" %in% names(fam)) "fam" else if ("FID" %in% names(fam)) "FID" else names(fam)[1L]
    id_col <- if ("id" %in% names(fam)) "id" else if ("IID" %in% names(fam)) "IID" else names(fam)[2L]
    fid <- clean_ids(fam[[fam_col]])
    iid <- clean_ids(fam[[id_col]])
    candidates[["IID"]] <- iid
    candidates[["FID:IID"]] <- paste(fid, iid, sep = ":")
  }
  for (candidate_name in names(candidates)) {
    ids <- candidates[[candidate_name]]
    if (anyDuplicated(ids) || !all(trait_ids %in% ids)) next
    index <- match(trait_ids, ids)
    out <- K[index, index, drop = FALSE]
    dimnames(out) <- list(trait_ids, trait_ids)
    asym <- max(abs(out - t(out)))
    scale <- max(1, max(abs(out)))
    if (asym > 1e-7 * scale)
      stop("GRM is not symmetric after alignment: ", label, call. = FALSE)
    return((out + t(out)) / 2)
  }
  available <- if (length(candidates)) unique(unlist(candidates, use.names = FALSE)) else character()
  missing <- setdiff(trait_ids, available)
  example <- paste(head(missing, 5L), collapse = ", ")
  stop("Could not align GRM '", label, "' to trait IDs. Missing examples: ", example,
       call. = FALSE)
}

read_grms <- function(prefixes, trait_ids, jitter) {
  if (!requireNamespace("genio", quietly = TRUE)) stop("Package 'genio' is not installed", call. = FALSE)
  lapply(prefixes, function(path) {
    prefix <- norm_grm_prefix(path)
    message("Reading GRM: ", prefix)
    obj <- genio::read_grm(prefix)
    fam <- if (!is.null(obj$fam)) obj$fam else NULL
    K <- align_grm(obj$kinship, fam, trait_ids, prefix)
    diag(K) <- diag(K) + jitter
    K
  })
}

empty_prediction <- function(traits) {
  out <- matrix(NA_real_, nrow(traits), ncol(traits),
                dimnames = list(rownames(traits), colnames(traits)))
  out
}

predict_rrblup <- function(K, traits, prediction) {
  if (!requireNamespace("rrBLUP", quietly = TRUE)) stop("Package 'rrBLUP' is not installed", call. = FALSE)
  out <- empty_prediction(traits)
  for (j in seq_len(ncol(traits))) {
    y <- as.numeric(traits[[j]])
    observed <- is.finite(y)
    if (sum(observed) < 3L) {
      warning("Skipping trait with fewer than three observations: ", colnames(traits)[j])
      next
    }
    if (var(y[observed]) <= .Machine$double.eps) {
      out[, j] <- if (prediction == "breeding_value") 0 else mean(y[observed])
      next
    }
    fit <- rrBLUP::mixed.solve(y = y, K = K, method = "REML", SE = FALSE)
    out[, j] <- if (prediction == "breeding_value") as.numeric(fit$u) else
      as.numeric(fit$beta[1L]) + as.numeric(fit$u)
  }
  out
}

sum_bglr_effects <- function(eta, n, p = NULL) {
  if (is.null(p)) {
    out <- numeric(n)
    for (term in eta) if (!is.null(term$u)) out <- out + as.numeric(term$u)
  } else {
    out <- matrix(0, n, p)
    for (term in eta) if (!is.null(term$u)) out <- out + as.matrix(term$u)
  }
  out
}

predict_bglr_univariate <- function(K_list, traits, opt) {
  if (!requireNamespace("BGLR", quietly = TRUE)) stop("Package 'BGLR' is not installed", call. = FALSE)
  n <- nrow(traits)
  out <- empty_prediction(traits)
  for (j in seq_len(ncol(traits))) {
    y <- as.numeric(traits[[j]])
    observed <- is.finite(y)
    if (sum(observed) < 3L) {
      warning("Skipping trait with fewer than three observations: ", colnames(traits)[j])
      next
    }
    if (var(y[observed]) <= .Machine$double.eps) {
      out[, j] <- if (opt$prediction == "breeding_value") 0 else mean(y[observed])
      next
    }
    set.seed(opt$seed + j - 1L)
    eta <- lapply(K_list, function(K) list(K = K, model = "RKHS"))
    prefix <- tempfile(pattern = paste0("bglr_trait_", j, "_"))
    fit <- BGLR::BGLR(y = y, ETA = eta, nIter = opt$nIter, burnIn = opt$burnIn,
                      thin = opt$thin, R2 = opt$R2, saveAt = prefix,
                      verbose = opt$verbose, rmExistingFiles = TRUE)
    unlink(Sys.glob(paste0(prefix, "*")), force = TRUE)
    bv <- sum_bglr_effects(fit$ETA, n)
    out[, j] <- if (opt$prediction == "breeding_value") bv else as.numeric(fit$mu) + bv
  }
  out
}

solve_spd <- function(A, b) {
  scale <- max(1, mean(diag(A)))
  for (ridge in scale * c(0, 1e-10, 1e-8, 1e-6)) {
    R <- tryCatch(chol(A + diag(ridge, nrow(A))), error = function(e) NULL)
    if (!is.null(R)) return(backsolve(R, forwardsolve(t(R), b)))
  }
  qr.solve(A, b, tol = 1e-10)
}

conditional_phenotype_mean <- function(Y, genetic_mean, R) {
  out <- genetic_mean
  patterns <- split(seq_len(nrow(Y)), apply(is.na(Y), 1L, paste0, collapse = ""))
  for (rows in patterns) {
    observed <- !is.na(Y[rows[1L], ])
    missing <- !observed
    if (!any(observed) || !any(missing)) next
    weights <- R[missing, observed, drop = FALSE] %*%
      solve_spd(R[observed, observed, drop = FALSE], diag(sum(observed)))
    residual <- Y[rows, observed, drop = FALSE] - genetic_mean[rows, observed, drop = FALSE]
    out[rows, missing] <- genetic_mean[rows, missing, drop = FALSE] + residual %*% t(weights)
  }
  out
}

predict_bglr_multitrait <- function(K_list, traits, opt) {
  if (!requireNamespace("BGLR", quietly = TRUE)) stop("Package 'BGLR' is not installed", call. = FALSE)
  n_observed <- colSums(!is.na(traits))
  trait_variance <- vapply(traits, var, numeric(1L), na.rm = TRUE)
  keep <- which(n_observed >= 3L & is.finite(trait_variance) &
                trait_variance > .Machine$double.eps)
  if (!length(keep)) stop("No trait has at least three observations", call. = FALSE)
  dropped <- setdiff(seq_len(ncol(traits)), keep)
  if (length(dropped)) warning("Skipping traits with fewer than three observations or zero variance: ",
                               paste(colnames(traits)[dropped], collapse = ", "))
  Y <- as.matrix(traits[, keep, drop = FALSE])
  n <- nrow(Y)
  eta <- lapply(K_list, function(K) list(K = K, model = "RKHS"))
  names(eta) <- paste0("GRM", seq_along(eta))
  residual_type <- if (opt$re == "DIAG") "DIAG" else "UN"
  set.seed(opt$seed)
  prefix <- tempfile(pattern = "bglr_multitrait_")
  fit <- BGLR::Multitrait(y = Y, ETA = eta, resCov = list(type = residual_type),
                          R2 = opt$R2, nIter = opt$nIter, burnIn = opt$burnIn,
                          thin = opt$thin, saveAt = prefix, verbose = opt$verbose)
  unlink(Sys.glob(paste0(prefix, "*")), force = TRUE)
  bv <- sum_bglr_effects(fit$ETA, n, ncol(Y))
  genetic_mean <- sweep(bv, 2L, as.numeric(fit$mu), "+")
  if (!is.null(fit$ETAHat)) {
    discrepancy <- max(abs(genetic_mean - as.matrix(fit$ETAHat)), na.rm = TRUE)
    if (is.finite(discrepancy) && discrepancy > 1e-6)
      warning("Manual genetic mean differs from fit$ETAHat; using fit$ETAHat")
    genetic_mean <- as.matrix(fit$ETAHat)
  }
  selected <- if (opt$prediction == "breeding_value") bv else genetic_mean
  if (opt$prediction == "phenotype") {
    if (is.null(fit$resCov$R)) stop("BGLR did not return fit$resCov$R", call. = FALSE)
    selected <- conditional_phenotype_mean(Y, genetic_mean, as.matrix(fit$resCov$R))
  }
  out <- empty_prediction(traits)
  out[, keep] <- selected
  out
}

write_predictions <- function(path, id_name, ids, prediction) {
  parent <- dirname(path)
  if (!dir.exists(parent)) dir.create(parent, recursive = TRUE, showWarnings = FALSE)
  out <- data.frame(ids, as.data.frame(prediction, check.names = FALSE),
                    check.names = FALSE, stringsAsFactors = FALSE)
  names(out)[1L] <- id_name
  write.csv(out, path, row.names = FALSE, quote = FALSE, na = "")
}

argv <- commandArgs(trailingOnly = TRUE)
if (!length(argv)) usage(1L)
parsed <- parse_args(argv)
opt <- parsed$opt
validate_options(opt, parsed$positional)

input <- read_traits_csv(parsed$positional[1L])
K_list <- read_grms(opt$grms, input$ids, opt$jitter)
message("Model: ", opt$model, " | GRMs: ", length(K_list), " | residual: ", opt$re,
        " | prediction: ", opt$prediction)
message("Animals: ", nrow(input$traits), " | traits: ", ncol(input$traits))

prediction <- if (opt$model == "rrblup") {
  predict_rrblup(K_list[[1L]], input$traits, opt$prediction)
} else if (opt$re == "0") {
  predict_bglr_univariate(K_list, input$traits, opt)
} else {
  predict_bglr_multitrait(K_list, input$traits, opt)
}

write_predictions(parsed$positional[2L], input$id_name, input$ids, prediction)
message("Wrote: ", parsed$positional[2L])
