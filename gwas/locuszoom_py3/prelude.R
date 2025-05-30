# prelude.R
print(paste("axisTextSize:", args[['axisTextSize']], "Type:", typeof(args[['axisTextSize']])))
print(paste("leftMarginLines:", args[['leftMarginLines']], "Type:", typeof(args[['leftMarginLines']])))
print(paste("rightMarginLines:", args[['rightMarginLines']], "Type:", typeof(args[['rightMarginLines']])))

# Ensure arguments are numeric
args[['axisTextSize']] <- as.numeric(args[['axisTextSize']])
args[['leftMarginLines']] <- as.numeric(args[['leftMarginLines']])
args[['rightMarginLines']] <- as.numeric(args[['rightMarginLines']])

# Print again after conversion
print(paste("Converted axisTextSize:", args[['axisTextSize']], "Type:", typeof(args[['axisTextSize']])))
print(paste("Converted leftMarginLines:", args[['leftMarginLines']], "Type:", typeof(args[['leftMarginLines']])))
print(paste("Converted rightMarginLines:", args[['rightMarginLines']], "Type:", typeof(args[['rightMarginLines']])))

