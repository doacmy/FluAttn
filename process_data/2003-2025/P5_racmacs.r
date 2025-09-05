# 使用Racmacs包构建抗原图谱并计算抗原距离矩阵
# 先设置工作路径 setwd("PATH_TO_YOUR_WORKING_DIRECTORY")
# 后运行源码 source("racmacs.r")

# Wilks S. Racmacs: Antigenic Cartography Macros. R package version 1.2.9. 2024.

library(Racmacs)
options(RacOptimizer.num_cores = 4)
path_to_titer_file <- "2003-2025_HI_matrix.csv"
titer_table        <- read.titerTable(path_to_titer_file)
map <- acmap(
  titer_table = titer_table
)
map <- optimizeMap(
  map                     = map,
  number_of_dimensions    = 2,
  number_of_optimizations = 100,
  minimum_column_basis    = "none"
)
coords <- agCoords(map)
distance_matrix <- as.matrix(dist(coords, method="euclidean"))
write.csv(distance_matrix, "2003_2025_distance_matrix.csv")

antigen_names <- agNames(map)
coord_df <- data.frame(
  short_name = antigen_names,
  MDS1 = coords[, 1],
  MDS2 = coords[, 2]
)
write.csv(coord_df, "2003_2025_antigen_coordinates.csv", row.names = FALSE)

