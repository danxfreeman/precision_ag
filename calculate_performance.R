# Load libraries.
library(jsonlite)

# Tabulate the top prediction and score for each image.
JSONtoDF <- function(json_path) {
  #
  # Args:
  #   json_path: Path to json file produced by test.py script.
  #
  # Returns:
  #   Dataframe.
  #
  # Convert json file to list objects.
  json <- fromJSON(json_path)
  images <- json$images$image
  classifiers <- json$images$classifiers
  
  # Create empty dataframe.
  df <- data.frame("parent" = NA, "image" = NA, "true_class" = NA,
                   "predicted_class" = NA, "score" = NA)
  
  # For each image:
  for(i in 1:length(images)) {
    # Identify image.
    path <- images[[i]]
    path <- strsplit(path, "/")[[1]]
    path <- rev(path)
    image <- path[1]
    zip <- path[2]
    class <- path[3]
    model <- path[4]
    # Get top prediction and score.
    all_scores <- classifiers[[i]]$classes[[1]]$score
    all_classes <- classifiers[[i]]$classes[[1]]$class
    score <- max(all_scores)
    prediction <- all_classes[all_scores == score]
    # Update dataframe.
    df[i,] <- c(model, image, class, prediction, score)
  }
  
  # Return dataframe.
  return(df)
  
}

# Calculate the precision, recall, and F1 score of each class.
calculatePerformance <- function(df, threshold) {
  #
  # Args:
  #   df: Dataframe produced by JSONtoDF function.
  #   threshold: Minimum score to be considered positive.
  #
  # Returns:
  #   Vector of performance metrics.
  #
  # Identify positives, true positives, false positives, and false negatives.
  df$P <- df$score > threshold
  df$TP <- df$P == T & df$predicted_class == df$true_class
  df$FP <- df$P == T & df$predicted_class != df$true_class
  df$FN <- df$P == F & df$predicted_class == df$true_class
  
  # Calculate performance.
  N <- sum(df$true_class != "negative")
  precision <- sum(df$TP) / sum(df$P) # true positives over predicted positives
  recall <- sum(df$TP) / sum(df$TP + df$FN) # true positives over all positives
  F1 <- (2*precision*recall)/(precision+recall)
  output <- c("N" = N, "precision" = precision, "recall" = recall, "F1" = F1)
  output[is.na(output)] <- 0
  
  # Return vector.
  return(output)
  
}

# Wrapper function concatenates the dataframe of each positive class to the 
# negative class and calculates performance. 
concatDF <- function(parent = "", negative, positive, threshold = 0.5) {
  #
  # Args:
  #   Parent (optional): Parent directory to which 'positive' and 'negative'
  #   paths will be concatenated. Must end with '/'.
  #   Negative: Path to the negative class json file, including file.
  #   Positive: Vector of paths to the positive class json files, including files.
  #   Threshold: Minimum score to be considered a positive.
  #
  # Returns:
  #   Dataframe.
  #
  # Concatenate negative and positive paths to the parent.
  full_negative = paste0(parent, negative)
  full_positive = paste0(parent, positive)
  #
  # Create negative class dataframe.
  neg_df <- JSONtoDF(full_negative)
  #
  # Create empty dataframe for storing results.
  df <- data.frame("parent" = NA, "class" = NA, "N" = NA, "precision" = NA,
                   "recall" = NA, "F1" = NA)
  #
  # For each positive class:
  for(i in 1:length(full_positive)) {
    # Create positive class dataframe.
    pos_path <- full_positive[i]
    pos_df <- JSONtoDF(pos_path)
    # Describe positive class.
    pos_class <- pos_df$true_class[1]
    pos_parent <- pos_df$parent[1]
    # Concatenate to negative class dataframe.
    concat_df <- rbind(pos_df, neg_df)
    # Calculate performance.
    performance <- calculatePerformance(concat_df, threshold)
    # Add class description and performance to dataframe.
    df[i,] <- c(pos_parent, pos_class, performance)
  }
  #
  # Return dataframe.
  return(df)
  
}