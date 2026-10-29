# ==============================================================================
# Virtual R Data Analyst Internship — Week 4 Final Capstone R Script
# Title: Comprehensive Data Analysis, Visualization & Predictive Modeling in R
# Author: Priyanshu Dubey
# Dataset: Motor Trend Car Road Tests (mtcars, N = 32, 11 Variables)
# ==============================================================================

# 1. Load Required Packages
library(datasets)
library(dplyr)
library(tidyr)
library(ggplot2)
library(psych)
library(car)
library(lmtest)
library(caret)
library(pROC)
library(glmnet)

data(mtcars)
stopifnot(nrow(mtcars) == 32, ncol(mtcars) == 11, sum(is.na(mtcars)) == 0)

# ==============================================================================
# PHASE I (WEEK 1): Data Cleaning, Group-Median Imputation, Outliers & Scaling
# ==============================================================================
set.seed(101)
mtcars_sim_na <- mtcars
mtcars_sim_na$hp[c(3, 10, 24)] <- NA  # Simulate 10% sensor dropout

mtcars_imputed <- mtcars_sim_na %>%
  group_by(cyl) %>%
  mutate(hp = ifelse(is.na(hp), median(hp, na.rm = TRUE), hp)) %>%
  ungroup()

cat("Original HP Mean:", mean(mtcars$hp), "| Imputed HP Mean:", mean(mtcars_imputed$hp), "\n")

# Tukey 1.5 x IQR Outlier Detection
detect_iqr_outliers <- function(x) {
  q <- quantile(x, probs = c(0.25, 0.75), na.rm = TRUE)
  iqr <- q[2] - q[1]
  which(x < (q[1] - 1.5 * iqr) | x > (q[2] + 1.5 * iqr))
}

for (v in c("mpg", "disp", "hp", "drat", "wt", "qsec", "carb")) {
  idx <- detect_iqr_outliers(mtcars[[v]])
  if (length(idx) > 0) {
    cat("IQR Outliers in", v, ":", paste(rownames(mtcars)[idx], collapse = ", "), "\n")
  }
}

# Normalization & Factor Encoding
min_max_norm <- function(x) (x - min(x)) / (max(x) - min(x))

mtcars_clean <- mtcars %>%
  mutate(
    model_name = rownames(mtcars),
    cyl_f  = factor(cyl, levels = c(4, 6, 8), labels = c("4-Cyl", "6-Cyl", "8-Cyl")),
    vs_f   = factor(vs, levels = c(0, 1), labels = c("V-Shaped", "Straight")),
    am_f   = factor(am, levels = c(0, 1), labels = c("Automatic", "Manual")),
    gear_f = factor(gear, levels = c(3, 4, 5), labels = c("3-Gear", "4-Gear", "5-Gear")),
    hp_per_wt = hp / wt,
    gpm_100   = 100 / mpg,
    across(c(mpg, disp, hp, drat, wt, qsec), list(z = ~as.numeric(scale(.)), norm = min_max_norm))
  )

# ==============================================================================
# PHASE II (WEEK 2): Exploratory Summary & Multi-Dimensional ggplot2 Visuals
# ==============================================================================
subgroup_summary <- mtcars_clean %>%
  group_by(cyl_f, am_f) %>%
  summarise(
    n = n(),
    Mean_MPG = round(mean(mpg), 2),
    SD_MPG   = round(sd(mpg), 2),
    Mean_WT  = round(mean(wt), 2),
    Mean_HP  = round(mean(hp), 1),
    .groups  = "drop"
  )
print(subgroup_summary)

# ==============================================================================
# PHASE III (WEEK 3): Statistical Hypothesis Testing & Predictive Modeling
# ==============================================================================
# 1. Shapiro-Wilk Normality Tests
sw_results <- sapply(mtcars[, c("mpg", "disp", "hp", "drat", "wt", "qsec")], function(x) {
  sw <- shapiro.test(x)
  c(W = unname(sw$statistic), p_value = sw$p.value)
})
print(round(t(sw_results), 4))

# 2. Correlation, Welch Two-Sample t-Test, One-Way ANOVA & Tukey HSD
print(cor.test(mtcars$wt, mtcars$mpg, method = "pearson"))
print(t.test(mpg ~ am, data = mtcars, var.equal = FALSE))
aov_cyl <- aov(mpg ~ factor(cyl), data = mtcars)
print(summary(aov_cyl))
print(TukeyHSD(aov_cyl))

# 3. Multiple Linear Regression & Interaction Modeling
fit_full  <- lm(mpg ~ ., data = mtcars)
fit_step  <- step(fit_full, direction = "both", trace = 0)
fit_hp    <- lm(mpg ~ wt + hp + am, data = mtcars)
fit_inter <- lm(mpg ~ wt * am + qsec, data = mtcars)

print(summary(fit_inter))
print(anova(fit_step, fit_inter))
print(car::vif(fit_step))
print(lmtest::bptest(fit_inter))
print(shapiro.test(resid(fit_inter)))

# 4. Cross-Validation (LOOCV & 10-Fold CV via caret)
set.seed(42)
ctrl_loocv <- trainControl(method = "LOOCV")
ctrl_cv10  <- trainControl(method = "cv", number = 10)
cv_inter_loo <- train(mpg ~ wt * am + qsec, data = mtcars, method = "lm", trControl = ctrl_loocv)
cv_inter_10f <- train(mpg ~ wt * am + qsec, data = mtcars, method = "lm", trControl = ctrl_cv10)
print(cv_inter_loo$results)
print(cv_inter_10f$results)

# 5. Binomial GLM Classification (Transmission am & Engine Architecture vs)
glm_am <- glm(am ~ mpg + qsec, data = mtcars, family = binomial(link = "logit"))
glm_vs <- glm(vs ~ wt + hp, data = mtcars, family = binomial(link = "logit"))
print(summary(glm_am))
print(summary(glm_vs))

# ==============================================================================
# PHASE IV (WEEK 4): Fleet Economic & Lightweighting Impact Simulation
# ==============================================================================
fleet_simulation <- subgroup_summary %>%
  mutate(
    Annual_Gallons_1000_Cars = round((1000 * 15000) / Mean_MPG, 0),
    Annual_Cost_USD_at_3.50  = round(Annual_Gallons_1000_Cars * 3.50, 0)
  )
print(fleet_simulation)
