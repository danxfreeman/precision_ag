# Load libraries.
library(spatstat)
library(tidyverse)

# Specify paths.
coord_path <- "/Users/danielfreeman/Desktop/ag/coord.csv" # csv file created by work_data.py
image_path <- "/Users/danielfreeman/Desktop/ag/Cropped/" # directory containing cropped images
to_path <- "/Users/danielfreeman/Desktop/ag/Split/" # directory where train and test sets will be stored

# Upload coordinates.
dat <- read.csv(coord_path)
colnames(dat) <- c("dataset", "image", "species", "condition", "x_start", "y_start", "width", "height", "id")

# Calculate the center of each bounding box.
dat$x_center <- dat$x_start + dat$width/2
dat$y_center <- dat$y_start + dat$height/2

# Temp.
dat <- dat[dat$dataset != 'FLT1_2',] # remove images from dataset 'FLT1_2'
dat <- dat[dat$condition != 'unknown',] # remove phosphorus stress images
dat <- dat[dat$condition != 'low_water_stress',] # remove phosphorus stress images
dat$dataset <- as.character(dat$dataset) # manually assign MAPIR images to FLT1 or FLT2
dat$dataset[dat$image %in% c("2017_0810_145123_112.JPG", "2017_0810_145127_114.JPG",
                           "2017_0810_145131_116.JPG", "2017_0810_145135_118.JPG",
                           "2017_0810_145139_120.JPG")] <- 'MAPIR_FLT1'
dat$dataset[dat$image %in% c("2017_0810_154133_084.JPG", "2017_0810_154137_086.JPG",
                           "2017_0810_154142_088.JPG")] <- 'MAPIR_FLT2'
# dat$condition <- as.character(dat$condition) # pool HWS and LWS images
# dat$condition[dat$condition == 'high_water_stress'] <- 'stress'
# dat$condition[dat$condition == 'low_water_stress'] <- 'stress'
# dat$species <- 'pool' # pool species

# Sort cropped images by position and assign to k-fold train and test sets.
df <- data.frame()
# Loop through each combination of species and condition.
dat$model <- paste(dat$dataset, dat$species, dat$condition, sep = "_")
for(model in split(dat, dat$model)) {
  # Normalize coordinates.
  model$x_norm <- (model$x_center - min(model$x_center)) / (max(model$x_center) - min(model$x_center))
  model$y_norm <- (model$y_center - min(model$y_center)) / (max(model$y_center) - min(model$y_center))
  # Select a reference image.
  ref_img <- model$image[1]
  ref <- model[model$image == ref_img,]
  ppp1 <- ppp(ref$x_norm, ref$y_norm)
  # Loop through each image.
  for(image in split(model, model$image)) {
    # Skip empty datafames.
    if(nrow(image) == 0) {next}
    # Report error and skip if number of points don't match.
    if(nrow(image) != nrow(ref)) {
      warning(
        "Number of points doesn't match \n",
        "Species:", as.character(ref$species[1]), "\n",
        "Condition:", as.character(ref$condition[1]), "\n",
        "Reference:", as.character(ref$image[1]), "\n",
        "Points:", nrow(ref), "\n",
        "Target:", as.character(image$image[1]), "\n",
        "Points:", nrow(image), "\n\n"
      )
      next
    }
    # Match points with reference image.
    ppp2 <- ppp(image$x_norm, image$y_norm)
    match <- pppdist(ppp1, ppp2)
    mx <- matrix(match)[3,][[1]]
    image$position <- apply(mx, 1, function(x) {which(x == 1)})
    # Sort cropped images by position.
    image <- image[order(image$position),]
    # Split observations into k-fold test and training sets.
    folds <- cut(seq(1:nrow(image)), breaks = 4, labels = F)
    for(k in 1:4) {
      image[paste0("train_k", k)] <- folds != k
      image[paste0("test_k", k)] <- folds == k
    }
    # Concatenate image dataframe.
    df <- rbind(df, image)
  }
}

# Append path to each cropped image.
files <- list.files(image_path, full.names = T, recursive = T)
for(i in 1:nrow(df)) {
  end <- paste0("_", df$id[i], ".JPG")[1]
  df$path[i] <- files[endsWith(files, end)][1]
}

# Create parent directory if none exists.
if(!dir.exists(to_path)) {
  dir.create(to_path)
}

# Create directory for each model.
setwd(to_path)
# Subset images to be used in each model.
for(model in unique(df$model)) {
  cat(paste("Splitting model:", model, "\n"))
  subset <- df[df$model == model,]
  # Sort by position so that related images appear together if sorted by date.
  subset <- subset[order(subset$position),]
  # Loop through each fold.
  for(k in 1:4) {
    # Define directory names.
    train_dir <- paste0(model, "_k", k, "_train")
    test_dir <- paste0(model, "_k", k, "_test")
    # Create directories.
    dir.create(train_dir)
    dir.create(test_dir)
    # Get indices of training and test set images.
    train_index <- subset[[paste0("train_k", k)]]
    test_index <- subset[[paste0("test_k", k)]]
    # Copy images.
    file.copy(subset$path[train_index], train_dir)
    file.copy(subset$path[test_index], test_dir)
    # Zip training and test sets.
    zip(zipfile = train_dir, files = train_dir)
    zip(zipfile = test_dir, files = test_dir)
  }
}

#View(df[,c(1:4,9,14:24)])
