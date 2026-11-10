# honestdid.R — Sensibilidad de Rambachan y Roth (2023) con el paquete oficial HonestDiD
# ---------------------------------------------------------------------------------------
# Insumos (los exporta 07_rambachan_roth.py):
#   resultados/tablas/honestdid_betahat.csv   coeficientes del estudio de eventos (pre e post), sin e = -1
#   resultados/tablas/honestdid_sigma.csv     matriz de varianzas (del bootstrap)
# Uso:   Rscript r/honestdid.R        (desde la raíz del repo)
# Instalar una vez:  install.packages("remotes"); remotes::install_github("asheshrambachan/HonestDiD")
suppressMessages(library(HonestDiD))

dir_t  <- file.path("resultados", "tablas")
beta   <- as.numeric(read.csv(file.path(dir_t, "honestdid_betahat.csv"))[1, ])
nombres <- names(read.csv(file.path(dir_t, "honestdid_betahat.csv")))
sigma  <- as.matrix(read.csv(file.path(dir_t, "honestdid_sigma.csv")))
n_pre  <- sum(grepl("^e-", nombres))          # coeficientes con e negativo
n_post <- length(beta) - n_pre
cat("Periodos previos:", n_pre, "| posteriores:", n_post, "\n")

# (a) Efecto inmediato (primer periodo posterior)
l_inm  <- basisVector(1, n_post)
# (b) Promedio de todos los periodos posteriores incluidos
l_prom <- matrix(1 / n_post, n_post, 1)

corre <- function(l_vec, etiqueta) {
  # Magnitudes relativas: la violación posterior es a lo más Mbar veces la mayor violación previa
  rm <- createSensitivityResults_relativeMagnitudes(
    betahat = beta, sigma = sigma, numPrePeriods = n_pre, numPostPeriods = n_post,
    bound = "deviation from parallel trends", method = "C-LF", Mbarvec = seq(0, 2, by = 0.25), l_vec = l_vec)
  # Suavidad: la pendiente de la tendencia diferencial cambia a lo más M entre periodos consecutivos
  sd <- createSensitivityResults(
    betahat = beta, sigma = sigma, numPrePeriods = n_pre, numPostPeriods = n_post,
    method = "FLCI", Mvec = seq(0, 0.02, length.out = 11), l_vec = l_vec)
  write.csv(rm, file.path(dir_t, paste0("honestdid_rm_", etiqueta, ".csv")), row.names = FALSE)
  write.csv(sd, file.path(dir_t, paste0("honestdid_sd_", etiqueta, ".csv")), row.names = FALSE)
  orig <- constructOriginalCS(betahat = beta, sigma = sigma, numPrePeriods = n_pre, numPostPeriods = n_post, l_vec = l_vec)
  g1 <- createSensitivityPlot_relativeMagnitudes(rm, orig)
  g2 <- createSensitivityPlot(sd, orig)
  ggplot2::ggsave(file.path("resultados", "figuras", paste0("honestdid_rm_", etiqueta, ".png")), g1, width = 8, height = 4.5, dpi = 200)
  ggplot2::ggsave(file.path("resultados", "figuras", paste0("honestdid_sd_", etiqueta, ".png")), g2, width = 8, height = 4.5, dpi = 200)
}
corre(l_inm,  "inmediato")
corre(l_prom, "promedio")
cat("Listo: tablas honestdid_{rm,sd}_*.csv y figuras honestdid_*.png\n")
