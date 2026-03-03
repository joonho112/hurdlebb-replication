## =============================================================================
## theme_manuscript.R -- Manuscript Plot Theme and Aesthetic Constants
## =============================================================================
## Purpose : Defines the custom ggplot2 theme and shared aesthetic constants
##           used across all manuscript figures and supplementary plots.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
##
## Requires : ggplot2 (>= 3.4.0)
## =============================================================================


###############################################################################
## Section 1 : Project Paths
###############################################################################

## Set project root (use here::here() if available, otherwise detect)
if (requireNamespace("here", quietly = TRUE)) {
    PROJECT_ROOT <- here::here()
} else {
    PROJECT_ROOT <- getwd()
}


###############################################################################
## Section 2 : Manuscript ggplot2 Theme
###############################################################################

#' Custom ggplot2 theme for manuscript figures
#'
#' Based on theme_bw() with lighter grid lines, borderless facet labels,
#' and bottom-positioned legends. Designed for print journal formatting.
#'
#' @param base_size Base font size (default: 10).
#' @return A ggplot2 theme object.

theme_manuscript <- function(base_size = 10) {
    ggplot2::theme_bw(base_size = base_size) +
        ggplot2::theme(
            ## Grid: lighter major lines, no minor
            panel.grid.minor  = ggplot2::element_blank(),
            panel.grid.major  = ggplot2::element_line(colour = "grey92",
                                                       linewidth = 0.3),
            ## Strip: borderless facet labels
            strip.background  = ggplot2::element_rect(fill = "grey96", colour = NA),
            strip.text        = ggplot2::element_text(face = "bold",
                                                       size = base_size),
            ## Axes: softer text, thin ticks
            axis.title        = ggplot2::element_text(size = base_size),
            axis.text         = ggplot2::element_text(size = base_size - 1,
                                                       colour = "grey25"),
            axis.ticks        = ggplot2::element_line(colour = "grey70",
                                                       linewidth = 0.3),
            ## Titles
            plot.title        = ggplot2::element_text(face = "bold",
                                                       size = base_size + 1,
                                                       hjust = 0),
            plot.subtitle     = ggplot2::element_text(size = base_size - 1,
                                                       hjust = 0,
                                                       colour = "grey40"),
            ## Legend
            legend.position   = "bottom",
            legend.text       = ggplot2::element_text(size = base_size - 1),
            legend.title      = ggplot2::element_text(size = base_size,
                                                       face = "bold"),
            legend.key.size   = grid::unit(0.8, "lines"),
            ## Margins
            plot.margin       = ggplot2::margin(5, 10, 5, 5)
        )
}


###############################################################################
## Section 3 : Shared Aesthetic Constants
###############################################################################

## -- Color Palettes -----------------------------------------------------------

## Margin color palette (extensive, intensive, reference, dispersion)
MARGIN_COLORS <- c(
    Extensive  = "#4393C3",   # blue
    Intensive  = "#D6604D",   # red
    Reference  = "#1B7837",   # green
    Dispersion = "#762A83"    # purple
)

## Reversal pattern palette (used in Figures F3, F4, F5)
REVERSAL_COLORS <- c(
    "Classic Reversal"  = "#D6604D",
    "Both Positive"     = "#4393C3",
    "Both Negative"     = "#762A83",
    "Opposite Reversal" = "#1B7837"
)


## -- Shape Encoding -----------------------------------------------------------

## Significance shape encoding
SHAPE_SIG    <- 16   # filled circle: CI excludes zero
SHAPE_NONSIG <-  1   # open circle:   CI includes zero


## -- Thresholds ---------------------------------------------------------------

## Small-state threshold (states with N < 40 observations)
SMALL_N_THRESHOLD <- 40


## -- geom_pointrange() Presets ------------------------------------------------
## ggplot2 4.0+: `size` controls point, `linewidth` controls line

POINTRANGE_DENSE <- list(linewidth = 0.35, size = 0.45)   # 51-state caterpillar
POINTRANGE_STD   <- list(linewidth = 0.7,  size = 1.25)   # standard coef plots


###############################################################################
## Section 4 : Figure and Table Save Helpers
###############################################################################

#' Save a ggplot figure in PDF and PNG formats
#'
#' @param plot A ggplot object.
#' @param name File name stem (without extension).
#' @param width  Width in inches (default: 7).
#' @param height Height in inches (default: 5).
#' @param output_dir Directory for output files. Defaults to
#'        PROJECT_ROOT/output/figures.

save_figure <- function(plot, name, width = 7, height = 5,
                        output_dir = file.path(PROJECT_ROOT, "output", "figures")) {

    if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

    pdf_path <- file.path(output_dir, paste0(name, ".pdf"))
    png_path <- file.path(output_dir, paste0(name, ".png"))

    ggplot2::ggsave(pdf_path, plot, width = width, height = height,
                    device = "pdf")
    ggplot2::ggsave(png_path, plot, width = width, height = height,
                    dpi = 300)

    cat(sprintf("  [SAVED] %s.pdf / .png (%g x %g in)\n", name, width, height))
}


#' Save a data frame as CSV and LaTeX table
#'
#' @param df A data frame.
#' @param name File name stem (without extension).
#' @param caption Table caption for LaTeX.
#' @param output_dir Directory for output files. Defaults to
#'        PROJECT_ROOT/output/tables.
#' @param ... Additional arguments passed to xtable::xtable().

save_table <- function(df, name, caption = "",
                       output_dir = file.path(PROJECT_ROOT, "output", "tables"),
                       ...) {

    if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

    csv_path <- file.path(output_dir, paste0(name, ".csv"))
    tex_path <- file.path(output_dir, paste0(name, ".tex"))

    write.csv(df, csv_path, row.names = FALSE)

    xt <- xtable::xtable(df, caption = caption, ...)
    print(xt, file = tex_path, include.rownames = FALSE,
          booktabs = TRUE, floating = FALSE)

    cat(sprintf("  [SAVED] %s.csv / .tex\n", name))
}
