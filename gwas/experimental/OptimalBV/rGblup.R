#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(genio)  # read_grm()
})

usage <- function() {
  cat(
"Usage:
  Rscript rrblup_s.R --model rrblup|bglr --grms grm1 [grm2 ...] --re 0|1 <traits_csv> <out_csv>

Flags:
  --model, -m   rrblup|bglr
  --grms,  -g   one or more GRM prefixes/files (space-separated)
  --re,    -r   0|1
               rrblup: only supports re=0 and exactly 1 GRM
               bglr  : re=0 => per-trait BGLR(); re=1 => Multitrait() with resCov

Optional:
  --nIter        (bglr only; default 6000)
  --burnIn       (bglr only; default 1500)
  --thin         (bglr only; default 1)
  --seed         (bglr only; default 1)
  --jitter       numeric; default 1e-8

Positional:
  <traits_csv>   CSV: first column = animal ID, remaining columns = traits (numeric; NA allowed)
  <out_csv>      output predictions CSV (same shape + order as input)
"
  )
}

# ----------------------------
# simple flag parser that supports: --grms g1 g2 g3 (space-separated until next flag)
# ----------------------------
parse_args <- function(args) {
  opt <- list(
    model = NULL,
    grms  = character(0),
    re    = 0L,
    nIter = 400L,
    burnIn = 200L,
    thin = 1L,
    seed = 1L,
    jitter = 1e-8,
    phenotype = FALSE
  )

  positional <- character(0)
  i <- 1L
  while (i <= length(args)) {
    tok <- args[i]

    if (tok %in% c("--help", "-h")) {
      usage()
      quit(status = 0)
    }

    if (tok %in% c("--model", "-m")) {
      if (i == length(args)) stop("Missing value after --model/-m")
      opt$model <- tolower(args[i + 1L])
      i <- i + 2L
      next
    }

    if (tok %in% c("--re", "-r")) {
      if (i == length(args)) stop("Missing value after --re/-r")
      opt$re <- args[i + 1L]
      i <- i + 2L
      next
    }

    if (tok == "--phenotype" || tok == "-p") {opt$phenotype <- TRUE; i <- i + 1L; next}
    if (tok == "--blup" || tok == "-b") {opt$phenotype <- FALSE; i <- i + 1L; next}  

    if (tok %in% c("--nIter", "--niter")) {
      opt$nIter <- as.integer(args[i + 1L]); i <- i + 2L; next
    }
    if (tok %in% c("--burnIn", "--burnin")) {
      opt$burnIn <- as.integer(args[i + 1L]); i <- i + 2L; next
    }
    if (tok == "--thin") {
      opt$thin <- as.integer(args[i + 1L]); i <- i + 2L; next
    }
    if (tok == "--seed") {
      opt$seed <- as.integer(args[i + 1L]); i <- i + 2L; next
    }
    if (tok == "--jitter") {
      opt$jitter <- as.numeric(args[i + 1L]); i <- i + 2L; next
    }

    if (tok %in% c("--grms", "-g")) {
      i <- i + 1L
      if (i > length(args)) stop("No GRM values after --grms/-g")
      while (i <= length(args) && !startsWith(args[i], "-")) {
        opt$grms <- c(opt$grms, args[i])
        i <- i + 1L
      }
      next
    }

    # otherwise: positional
    positional <- c(positional, tok)
    i <- i + 1L
  }

  list(opt = opt, positional = positional)
}

# ----------------------------
# normalize GRM prefix (allow passing .../foo, .../foo.grm, .../foo.grm.bin)
# ----------------------------
norm_grm_prefix <- function(x) {
  x <- sub("\\.grm\\.bin$", "", x, ignore.case = TRUE)
  x <- sub("\\.grm\\.N\\.bin$", "", x, ignore.case = TRUE)
  x <- sub("\\.grm\\.id$", "", x, ignore.case = TRUE)
  x <- sub("\\.grm$", "", x, ignore.case = TRUE)
  x
}

# ----------------------------
# read traits: first column is ID, force to character to preserve 0012 etc.
# ----------------------------
read_traits_csv <- function(path) {
  if (!file.exists(path)) stop("Missing traits file: ", path)
  raw <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  if (ncol(raw) < 2) stop("Trait CSV must have >=2 columns: ID + >=1 trait")
  ids <- as.character(raw[[1]])
  df <- raw[-1]
  rownames(df) <- ids
  for (j in seq_len(ncol(df))) df[[j]] <- suppressWarnings(as.numeric(df[[j]]))
  df
}

clean_ids <- function(ids) {
  # Force to character, trimming whitespace
  ids <- trimws(as.character(ids))
  # Remove scientific notation (e.g., 9.33e+14 -> 933000...) if it exists
  ids <- format(ids, scientific = FALSE, trim = TRUE)
  return(ids)
}

# ----------------------------
# Align a GRM to trait IDs (tries rownames first; then uses fam table if needed)
# ----------------------------
align_grm <- function(K, fam_tbl, trait_ids) {
  # helper
  try_match <- function(ids) {
    ids <- as.character(ids)
    if (!all(trait_ids %in% ids)) return(NULL)
    K2 <- K[trait_ids, trait_ids, drop = FALSE]
    K2
  }

  # Case 1: K already has rownames
  if (!is.null(rownames(K))) {
    K2 <- try_match(rownames(K))
    if (!is.null(K2)) return(K2)
  }

  # Case 2: use fam table (genio::read_grm)
  if (!is.null(fam_tbl)) {
    # genio uses columns fam, id; but be robust
    fam_col <- if ("fam" %in% names(fam_tbl)) "fam" else if ("FID" %in% names(fam_tbl)) "FID" else names(fam_tbl)[1]
    id_col  <- if ("id"  %in% names(fam_tbl)) "id"  else if ("IID" %in% names(fam_tbl)) "IID" else names(fam_tbl)[2]

    fid <- as.character(fam_tbl[[fam_col]])
    iid <- as.character(fam_tbl[[id_col]])

    fid_iid <- paste(fid, iid, sep=":")
    # try FID:IID
    rownames(K) <- fid_iid; colnames(K) <- fid_iid
    K2 <- try_match(fid_iid)
    if (!is.null(K2)) return(K2)

    # try IID if unique
    if (length(unique(iid)) == length(iid)) {
      rownames(K) <- iid; colnames(K) <- iid
      K2 <- try_match(iid)
      if (!is.null(K2)) return(K2)
    }
  }

  stop("Could not align GRM to trait IDs. Example trait ID: ", trait_ids[1])
}

# ----------------------------
# rrBLUP: loop traits
# ----------------------------
predict_rrblup <- function(K, traits, jitter = 1e-8) {
  if (!requireNamespace("rrBLUP", quietly = TRUE)) stop("Package rrBLUP not installed.")
  n <- nrow(traits)
  K <- K + diag(jitter, n)

  preds <- matrix(NA_real_, n, ncol(traits))
  rownames(preds) <- rownames(traits)
  colnames(preds) <- colnames(traits)

  for (j in seq_len(ncol(traits))) {
    y <- as.numeric(traits[[j]])
    if (sum(!is.na(y)) < 3) next
    fit <- rrBLUP::mixed.solve(y = y, K = K, SE = FALSE)
    preds[, j] <- as.numeric(fit$beta) + as.numeric(fit$u)
  }
  as.data.frame(preds, check.names = FALSE, stringsAsFactors = FALSE)
}

# ----------------------------
# BGLR univariate: loop traits, multiple GRMs => multiple RKHS ETA terms
# ----------------------------
predict_bglr_univariate <- function(K_list, traits, jitter = 1e-8, nIter=6000, burnIn=1500, thin=1, seed=1) {
  if (!requireNamespace("BGLR", quietly = TRUE)) stop("Package BGLR not installed.")
  set.seed(seed)
  n <- nrow(traits)

  ETA <- lapply(K_list, function(K) list(K = K + diag(jitter, n), model="RKHS"))

  preds <- matrix(NA_real_, n, ncol(traits))
  rownames(preds) <- rownames(traits)
  colnames(preds) <- colnames(traits)

  for (j in seq_len(ncol(traits))) {
    y <- as.numeric(traits[[j]])
    if (sum(!is.na(y)) < 3) next
    fit <- BGLR::BGLR(y = y, ETA = ETA, nIter=nIter, burnIn=burnIn, thin=thin,
                     verbose=FALSE, rmExistingFiles=TRUE, saveAt="bglr_temp_files")
    # preds[, j] <- as.numeric(fit$yHat)
    yhat <- fit$yHat
    if (is.null(yhat) || (all(yhat == 0) && var(y, na.rm=TRUE) > 0)) {
        # Get Intercept (mu)
        mu <- if (is.null(fit$mu)) 0 else fit$mu
        # Sum Random Effects (u) from all GRMs
        u_total <- rep(0, n)
        for(k in seq_along(fit$ETA)) {
            if (!is.null(fit$ETA[[k]]$u)) {
                u_total <- u_total + fit$ETA[[k]]$u}
        }
        yhat <- mu + u_total}
    preds[, j] <- as.numeric(yhat)  
  }
  as.data.frame(preds, check.names = FALSE, stringsAsFactors = FALSE)
}

# ----------------------------
# BGLR multitrait with residual covariance: one model for all traits
# ----------------------------
# make_resCov <- function(re_flag, Y, eps_scale = 1e-6) {
#   re_flag <- toupper(as.character(re_flag))
#   t <- ncol(Y)
#   if (re_flag == "1") return(NULL)
#   # A safe positive-definite diagonal scale for priors
#   v <- apply(Y, 2, var, na.rm = TRUE)
#   v[!is.finite(v) | v <= 0] <- 1.0
#   eps <- eps_scale * median(v)
#   S0 <- diag(v + eps, t, t)
#   if (re_flag == "UN") {return(list(type = "UN", df0 = max(5L, t + 2L), S0 = S0)) }
#   if (re_flag == "UN") {
#       df0 <- max(t + 5L, 10L)    
#       S0  <- S0 * (df0 - t - 1L)
#       return(list(type = "UN", df0 = df0, S0 = S0))}
#   if (re_flag == "DIAG") {return(list(type = "DIAG"))}  # simplest + robust}
#   if (re_flag == "FA") {
#     M <- min(2L, t - 1L) 
#     return(list(type = "FA", M = M))
#   }
#   stop("--re must be one of: 1, UN, FA, DIAG")
# }



impute_phenotypes_conditional <- function(Y_obs, Y_genetic, R_cov) {
  # Y_obs: Original data with NAs
  # Y_genetic: The (mu + u) matrix we calculated manually
  # R_cov: The residual covariance matrix from BGLR (fit$resCov)
  n <- nrow(Y_obs)
  p <- ncol(Y_obs)
  Y_imputed <- Y_genetic # Start with genetic base
  for (i in 1:n) {
    y_i <- Y_obs[i, ]
    is_obs <- !is.na(y_i)
    is_miss <- is.na(y_i)
    
    # We can only impute deviations if we have:
    # 1. At least one observed trait
    # 2. At least one missing trait
    if (any(is_obs) && any(is_miss)) {
      
      # Partition the Residual Matrix
      # R_mo: Covariance between Missing and Observed
      # R_oo: Covariance between Observed and Observed
      R_mo <- R_cov[is_miss, is_obs, drop = FALSE]
      R_oo <- R_cov[is_obs, is_obs, drop = FALSE]
      
      # Calculate Residuals of Observed traits
      # (Observed Phenotype - Genetic Prediction)
      e_obs <- y_i[is_obs] - Y_genetic[i, is_obs]
      
      # Calculate Conditional Expected Residuals for Missing traits
      # E(e_miss | e_obs) = R_mo * inv(R_oo) * e_obs
      # We use 'solve' for matrix inversion
      e_miss_pred <- R_mo %*% solve(R_oo, e_obs)
      
      # Add this conditional environmental noise to the genetic prediction
      Y_imputed[i, is_miss] <- Y_imputed[i, is_miss] + as.numeric(e_miss_pred)
    }
  }
  return(Y_imputed)
}

make_residual_prior <- function(re_flag, Y) {
  re_flag <- toupper(as.character(re_flag))
  if (re_flag == "1") return(NULL)
  
  t <- ncol(Y)
  v <- apply(Y, 2, var, na.rm = TRUE)
  v[!is.finite(v) | v <= 1e-8] <- 1.0 
  
  target_var <- v * 0.5 
  S0_base <- diag(target_var, t, t)
  
  if (re_flag == "UN") {
    # UN requires a MATRIX S0
    df0 <- max(t + 6L, 10L)
    S0 <- S0_base * (df0 - t - 1)
    return(list(type = "UN", df0 = df0, S0 = S0))
  }
  
  if (re_flag == "DIAG") {
    # FIX: DIAG requires a VECTOR S0 (extract diagonal)
    S0_vector <- diag(S0_base) 
    return(list(type="DIAG", df0=5, S0=S0_vector))
  }
  
  if (re_flag == "FA") {
    return(list(type="FA", M=min(2,t-1), S0=S0_base))
  }
  stop("Invalid --re flag")
}

# --- 2. Helper for Genetics (The "G" Matrix) ---
make_genetic_prior <- function(Y, min_h2_guess = 0.05) {
  t <- ncol(Y)
  v <- apply(Y, 2, var, na.rm = TRUE)
  v[!is.finite(v) | v <= 1e-8] <- 1.0 
  # Target: Genetic Variance = 5% of Phenotypic Variance
  target_var <- v * min_h2_guess
  S0_base <- diag(target_var, t, t)
  # UN requires a MATRIX S0
  df0 <- max(t + 6L, 10L)
  scale_factor <- df0 - t - 1
  # Safety against invalid scale factor if t is huge
  if (scale_factor <= 0) scale_factor <- 1 
  S0 <- S0_base * scale_factor
  return(list(type = "UN", df0 = df0, S0 = S0))
}

# --- 3. The Main Function ---
predict_bglr_multitrait <- function(K_list, traits, jitter=1e-8, nIter=6000, burnIn=1500, thin=1, seed=1, phenotype=FALSE) {
  if (!requireNamespace("BGLR", quietly=TRUE)) stop("BGLR not installed.")
  set.seed(seed)
  
  keep <- which(colSums(!is.na(traits)) >= 3)
  if (length(keep)==0) stop("No sufficient data.")
  Y <- as.matrix(traits[, keep, drop=FALSE])
  n <- nrow(Y)

  # --- STEP 1: Construct Genetic Prior ---
  # This enforces your 5% heritability threshold
  genetic_prior <- make_genetic_prior(Y, min_h2_guess = 0.05)

  ETA <- lapply(K_list, function(K) {
      list(
          K = K + diag(jitter, n), 
          model = "RKHS",
          Cov = genetic_prior 
      )
  })

  # --- STEP 2: Run BGLR ---
  # We use the fixed residual prior (handling DIAG vector correctly)
  fit <- BGLR::Multitrait(
    y = Y,
    ETA = ETA,
    resCov = make_residual_prior(opt$re, Y), 
    nIter = nIter, burnIn = burnIn, thin = thin, verbose = FALSE,
    saveAt = "bglr_temp_files"
  )

  # --- Cleanup & Output ---
  yhat_genetic <- fit$yHat
  
  # Manual Reconstruction Fallback
  if (is.null(yhat_genetic)) {
     message("Warning: fit$yHat is NULL. Reconstructing...")
     yhat_genetic <- matrix(rep(fit$mu, each=n), nrow=n, ncol=ncol(Y))
     for(k in seq_along(fit$ETA)) if(!is.null(fit$ETA[[k]]$u)) yhat_genetic <- yhat_genetic + fit$ETA[[k]]$u
  }
  if (is.null(yhat_genetic)) stop("Failed to recover genetic predictions.")
  
  # Imputation
  final_preds <- yhat_genetic
  if (phenotype && !is.null(fit$resCov)) {
      message("Performing phenotypic imputation...")
      final_preds <- impute_phenotypes_conditional(Y, yhat_genetic, fit$resCov)
  }
  
  preds <- matrix(NA, n, ncol(traits))
  colnames(preds) <- colnames(traits)
  preds[, keep] <- final_preds
  as.data.frame(preds)
}

# predict_bglr_multitrait <- function(K_list, traits, jitter = 1e-8, nIter=6000, burnIn=1500, thin=1, seed=1, phenotype =FALSE) {
#   if (!requireNamespace("BGLR", quietly = TRUE)) stop("Package BGLR not installed.")
#   set.seed(seed)
#   n <- nrow(traits)
#   p <- ncol(traits)

#   # Drop traits with too few observed values to avoid model failures, reinsert as NA later
#   keep <- which(colSums(!is.na(traits)) >= 3)
#   drop <- setdiff(seq_len(p), keep)

#   if (length(keep) == 0) stop("No trait has >=3 observed values; cannot fit Multitrait().")

#   Y <- as.matrix(traits[, keep, drop=FALSE])

#   # ETA <- lapply(K_list, function(K) list(K = K + diag(jitter, n), model="RKHS"))
#   genetic_prior <- make_genetic_prior( Y)
#   ETA <- lapply(K_list, function(K) {
#       list( K = K + diag(jitter, n), 
#           model = "RKHS",Cov = genetic_prior)})

#   fit <- BGLR::Multitrait(
#     y = Y,
#     ETA = ETA,
#     resCov = make_resCov(opt$re, Y),
#     nIter = nIter,
#     burnIn = burnIn,
#     thin = thin,
#     verbose=TRUE,
#     saveAt = "bglr_temp_files",
#   )

#   # FALLBACK: Manually calculate yHat if missing
#   if (is.null(fit$yHat)) {
#     message("Warning: fit$yHat is NULL. Attempting manual reconstruction from fit$mu and fit$ETA...")
    
#     if (!is.null(fit$mu) && !is.null(fit$ETA)) {
#       # Initialize with Intercepts (mu)
#       # fit$mu is usually a vector of length p (one intercept per trait)
#       yHat_manual <- matrix(rep(fit$mu, each = n), nrow = n, ncol = length(keep))
      
#       # Add Random Effects (u) from all kernels
#       for (k in seq_along(fit$ETA)) {
#         if (!is.null(fit$ETA[[k]]$u)) { yHat_manual <- yHat_manual + fit$ETA[[k]]$u}
#       }
#       # return predicted trait if requested
#       if (phenotype) {
#         if (is.null(fit$resCov) || !is.matrix(fit$resCov)) {
#           warning("Cannot perform phenotypic imputation: fit$resCov is missing or not a matrix.")} 
#         else {
#           message("Performing conditional phenotypic imputation using Residual Covariance...")
#           yHat_manual <- impute_phenotypes_conditional(Y, yHat_manual, fit$resCov)}
#       } 
#       fit$yHat <- yHat_manual
#       message("Manual reconstruction successful.")
#     }
#   }

#   if (is.null(fit$yHat)) stop("Multitrait() output did not contain yHat; inspect fit object names().")

#   yhat <- fit$yHat
#   if (!is.matrix(yhat)) stop("Expected Multitrait() yHat to be a matrix.")

#   preds <- matrix(NA_real_, n, p)
#   rownames(preds) <- rownames(traits)
#   colnames(preds) <- colnames(traits)
#   preds[, keep] <- yhat
#   # dropped traits remain NA

#   as.data.frame(preds, check.names = FALSE, stringsAsFactors = FALSE)
# }

# ----------------------------
# MAIN
# ----------------------------
argv <- commandArgs(trailingOnly = TRUE)
parsed <- parse_args(argv)
opt <- parsed$opt
pos <- parsed$positional

if (is.null(opt$model) || length(opt$grms) == 0 || length(pos) != 2) {
  usage()
  stop("Missing required arguments. Need: --model, --grms, --re, and <traits_csv> <out_csv>.")
}

if (!(opt$model %in% c("rrblup", "bglr"))) stop("Invalid --model. Use rrblup or bglr.")
if (!(opt$re %in% c('0', '1', 'UN', 'FA', 'DIAG'))) stop("Invalid --re. Use 0 or 1.")

traits_csv <- pos[1]
out_csv <- pos[2]

traits <- read_traits_csv(traits_csv)
trait_ids <- rownames(traits)
# --- FIX: Sanitize Phenotype IDs ---
clean_row_names <- clean_ids(rownames(traits))
rownames(traits) <- clean_row_names
trait_ids <- clean_row_names

# read+align all GRMs
K_list <- list()
# for (g in opt$grms) {
#   prefix <- norm_grm_prefix(g)
#   message("Reading GRM: ", prefix)
#   obj <- genio::read_grm(prefix)
#   K <- obj$kinship
#   fam_tbl <- if (!is.null(obj$fam)) obj$fam else NULL
#   K_aligned <- align_grm(K, fam_tbl, trait_ids)
#   K_list[[length(K_list) + 1]] <- K_aligned
# }
for (g in opt$grms) {
  prefix <- norm_grm_prefix(g)
  message("Reading GRM: ", prefix)
  obj <- genio::read_grm(prefix)
  K <- obj$kinship
  # --- FIX: Sanitize GRM IDs ---
  # genio usually puts IDs in rownames/colnames. Clean them.
  if (!is.null(rownames(K))) {
      cleaned_k_ids <- clean_ids(rownames(K))
      rownames(K) <- cleaned_k_ids
      colnames(K) <- cleaned_k_ids}
  K_aligned <- align_grm(K, fam_tbl, trait_ids)
  # Check for NAs immediately after alignment
  if (any(is.na(K_aligned))) {
      stop(paste("CRITICAL: GRM alignment generated NAs for", prefix, 
                 ". This means the IDs in the CSV did not match the IDs in the GRM file."))
  }
  K_list[[length(K_list) + 1]] <- K_aligned
}                

# enforce rrBLUP constraints
if (opt$model == "rrblup") {
  if (length(K_list) != 1) stop("rrblup only supports exactly 1 GRM (you provided ", length(K_list), ").")
  if (opt$re != '0') stop("rrblup does not support residual/environmental covariance: require --re 0.")
}

message("Model: ", opt$model, " | #GRMs: ", length(K_list), " | re: ", opt$re)
message("Traits: n=", nrow(traits), " p=", ncol(traits))
# --- DEBUG: SANITY CHECK START ---
message("\n--- DATA INSPECTION ---")
# 1. Check Phenotypes
message("First 5 phenotype rows (Traits):")
print(head(traits[, 1:min(5, ncol(traits))]))
message("Phenotype Variance (first trait): ", var(traits[[1]], na.rm=TRUE))

# 2. Check GRM
message("GRM Dimensions: ", nrow(K_list[[1]]), " x ", ncol(K_list[[1]]))
message("First 5 GRM IDs: ", paste(rownames(K_list[[1]])[1:5], collapse=", "))
message("First 5 GRM Diagonal values (should be ~1.0):")
print(diag(K_list[[1]])[1:5])

# 3. Check Alignment strictly
if (!all(rownames(traits) == rownames(K_list[[1]]))) {
    stop("CRITICAL ERROR: GRM and Trait rownames do not match identically!")
} else {
    message("Alignment Check: IDs match identically.")
}

d <- diag(as.matrix(K_list[[1]]))
message("diag length = ", length(d),
        " | anyNA = ", anyNA(d),
        " | first5 = ", paste(format(d[1:5], digits=6), collapse=", "))
message("traits head rownames: ", paste(head(rownames(traits)), collapse=", "))
message("traits first row first5: ", paste(format(as.numeric(traits[1, 1:5])), collapse=", "))
message("-----------------------\n")

# --- DEBUG: SANITY CHECK END ---

if (opt$model == "rrblup") {
  pred <- predict_rrblup(K_list[[1]], traits, jitter = opt$jitter)
} else {
  if (opt$re == '0') {
    pred <- predict_bglr_univariate(K_list, traits, jitter = opt$jitter,
                                   nIter = opt$nIter, burnIn = opt$burnIn, thin = opt$thin, seed = opt$seed)
  } else {
    pred <- predict_bglr_multitrait(K_list, traits, jitter = opt$jitter, phenotype = opt$phenotype,
                                   nIter = opt$nIter, burnIn = opt$burnIn, thin = opt$thin, seed = opt$seed)
  }
}

message("Writing: ", out_csv)
write.csv(pred, out_csv, row.names = TRUE, quote = FALSE)
message("Done.")
