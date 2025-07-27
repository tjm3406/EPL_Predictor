# install.packages('caret')
# Load libraries
library(tidyverse)
library(glmnet)
library(caret)
library(lubridate)


# This project uses the premier league kaggle dataset of games from 2000-2018 to 
# predict who will win the game. It uses a binomial logistic regression classifier
# as well as linear regression to predict goal difference between two teams on a given
# matchday. Using this goal difference prediction I can predict who will win as well
# because if goal difference is positive than home team wins, and negative away team
# wins.
# 
# This project implements ideas such as elastic net for feature selection. As well
# as checks correlation matrix to determine collinearity. For feature selection,
# p-values are also checked to determine statistical significance, this is done after
# elastic net filters features. Combining all those methods allows for excellent
# feature selection. Graphs to determine feature importance are included throughout.
# 
# Accuracy score is used to determine quality of models.


data <- read_csv("C:/Users/Tyler/OneDrive/Documents/METCS555/Project/raw_dataset.csv")

glimpse(data)
summary(data)

data = data %>% mutate(FTR = case_when(FTR == 'H' ~ 1, FTR == 'NH' ~ 0))

table(data$FTR)

data <- data %>%
  mutate(Date = as.Date(Date, format = "%d/%m/%y"))
str(data$Date)

data <- data %>% arrange(Date)
data$HTHWinPct <- NA

# Loop to set a Head to Head win percentage feature
for (i in 1:nrow(data)) {
  current_date <- data$Date[i]
  home_team <- data$HomeTeam[i]
  away_team <- data$AwayTeam[i]
  
  # Filter previous matches between the two teams
  past_matches <- data %>%
    filter(Date < current_date &
             Date >= current_date - lubridate::years(3) &
             ((HomeTeam == home_team & AwayTeam == away_team) |
                (HomeTeam == away_team & AwayTeam == home_team)))
  
  if (nrow(past_matches) > 0) {
    # Count home team wins in head-to-head matches
    home_wins <- sum((past_matches$HomeTeam == home_team & past_matches$FTR == 1) |
                       (past_matches$AwayTeam == home_team & past_matches$FTR == 0))
    
    data$HTHWinPct[i] <- home_wins / nrow(past_matches)
  } else {
    data$HTHWinPct[i] <- 0.5 # Neutral number if games haven't been played between two teams yet
  }
}

# Remove non-numeric / non-useful columns if necessary (e.g., match date, team names)
data_model <- data %>%
  select(-Date, -HomeTeam, -AwayTeam, -FTHG, -FTAG, -ATFormPtsStr, -HTFormPtsStr, -MW,
         -HM1, -HM2, -HM3, -HM4, -HM5, -AM1, -AM2, -AM3, -AM4, -AM5) %>% 
  drop_na()

write.csv(data_model, "C:/Users/Tyler/OneDrive/Documents/METCS555/Project/Cleaned_premierleague.csv")

train_index <- createDataPartition(data_model$FTR, p = 0.7, list = FALSE)
train_data <- data_model[train_index, ]
test_data  <- data_model[-train_index, ]

# Ensure FTR is numeric
data_model$FTR <- as.numeric(data_model$FTR)

# Create model matrices
X <- model.matrix(FTR ~ . -1, data = train_data)
y <- train_data$FTR

X_test <- model.matrix(FTR ~ . -1, data = test_data)
y_test <- test_data$FTR

# Convert X_train to data frame and attach response for glm()
train_df <- as.data.frame(X)
train_df$FTR <- y

# Also do the same for test set if you want to use it later
test_df <- as.data.frame(X_test)
test_df$FTR <- y_test

set.seed(555)

# alpha = 0.5
cv_model <- cv.glmnet(X, y, alpha = 0.5, family = "binomial")  

# Plot CV error vs lambda
plot(cv_model)

# Best lambda
best_lambda <- cv_model$lambda.min
print(best_lambda)

coef_df <- coef(cv_model, s = "lambda.min") %>% 
  as.matrix() %>%
  as.data.frame() %>%
  rownames_to_column("feature") %>%
  rename(coef = s1) %>%
  filter(coef != 0 & feature != "(Intercept)") %>%  # Remove intercept and zeroed features
  mutate(abs_coef = abs(coef)) %>%
  arrange(desc(abs_coef))

# Plot
ggplot(coef_df, aes(x = reorder(feature, abs_coef), y = coef, fill = coef > 0)) +
  geom_col() +
  coord_flip() +
  labs(title = "Feature Importance (Elastic Net)",
       x = "Feature",
       y = "Coefficient",
       fill = "Effect on Home Win") +
  scale_fill_manual(values = c("TRUE" = "darkgreen", "FALSE" = "firebrick")) +
  theme_minimal()

selected_features <- coef_df$feature
selected_features
# Create formula string
glm_formula <- as.formula(
  paste("FTR ~", paste(selected_features, collapse = " + "))
)
glm_formula
# Fit logistic regression
glm_model <- glm(glm_formula, data = train_data, family = "binomial")
summary(glm_model)

pred_probs <- predict(glm_model, newdata = test_df, type = "response")

pred_class <- ifelse(pred_probs > 0.5, 1, 0)

actual <- test_df$FTR

accuracy <- mean(pred_class == actual)
print(paste("Accuracy:", round(accuracy, 4)))

removed_vars = c('HTGC', 'ATGS', 'HTP', 'ATWinStreak3', 'ATLossStreak3',
                 'HTWinStreak5', 'HTLossStreak3', 'ATGC', 'HTGS')
reduced_features <- setdiff(selected_features, removed_vars)

# Rebuild formula without removed features
glm_formula_reduced <- as.formula(
  paste("FTR ~", paste(reduced_features, collapse = " + "))
)

# Checking for collinearity
# install.packages('corrplot')
library(corrplot)
selected_features <- all.vars(glm_formula)[-1]
data_selected <- data_model %>% select(all_of(selected_features))
cor_matrix <- cor(data_selected, use = "complete.obs")
corrplot(cor_matrix, method = "color", type = "full", addCoef.col = 'black', number.cex = .6, tl.cex = 0.8, tl.col = "black")

glm_model <- glm(glm_formula_reduced, data = train_data, family = "binomial")
glm_summary = summary(glm_model)
glm_summary

pred_probs <- predict(glm_model, newdata = test_df, type = "response")

pred_class <- ifelse(pred_probs > 0.5, 1, 0)

actual <- test_df$FTR
accuracy <- mean(pred_class == actual)
print(paste("Accuracy:", round(accuracy, 4)))

library(caret)

confusionMatrix(factor(pred_class), factor(actual))

coef_df <- as.data.frame(glm_summary$coefficients)
coef_df <- coef_df %>%
  rownames_to_column(var = "Feature") %>%
  rename(
    Estimate = Estimate,
    StdError = `Std. Error`,
    Zvalue = `z value`,
    Pvalue = `Pr(>|z|)`
  )

library(ggplot2)

coef_df %>%
  filter(Feature != "(Intercept)") %>%
  ggplot(aes(x = reorder(Feature, -log10(Pvalue)), y = -log10(Pvalue))) +
  geom_col(fill = "skyblue") +
  coord_flip() +
  labs(
    title = "Feature Importance from GLM (by P-value)",
    x = "Feature",
    y = "-log10(P-value)"
  ) +
  theme_minimal()

# Overall, the quality of the model was not great. It had quite a low accuracy score
# and did not produce good results.
# 
# Next is part 2 where we use linear regression to predict goal difference and thus,
# predict the winner.


data_model <- data %>%
  mutate(GoalDiff = FTHG - FTAG)

data_model <- data_model %>%
  select(-FTHG, -FTAG, -FTR, -Date, -HomeTeam, -AwayTeam, -FTHG, -FTAG, -ATFormPtsStr, -HTFormPtsStr, -MW,
         -HM1, -HM2, -HM3, -HM4, -HM5, -AM1, -AM2, -AM3, -AM4, -AM5, -ATWinStreak3, -ATLossStreak3, -ATLossStreak5,
         -HTWinStreak5, -HTLossStreak3, -HTWinStreak3, -ATWinStreak5, -HTGC, -ATGC, -HTP, -ATP, -HTGS, -ATGS) %>% 
  drop_na()

glimpse(data_model)

train_index <- createDataPartition(data_model$GoalDiff, p = 0.7, list = FALSE)
train_data <- data_model[train_index, ]
test_data <- data_model[-train_index, ]

lm_model <- lm(GoalDiff ~ ., data = train_data)
summary(lm_model)

test_preds <- predict(lm_model, newdata = test_data)

# 1 = Home Win, 0 = Not Home Win
predicted_win <- ifelse(test_preds > 0, 1, 0)

real_win <- ifelse(test_data$GoalDiff > 0, 1, 0)

predicted_win <- factor(predicted_win, levels = c(0,1))
real_win <- factor(real_win, levels = c(0,1))

confusionMatrix(predicted_win, real_win)

# Overall, the quality of the model was even worse. It had a lower accuracy score
# and did not produce good results.
# 
# While this was a clever linear regression classification attempt, it did not
# perform well.
#
# Future ideas for this project include adding other features that provide better
# results.
