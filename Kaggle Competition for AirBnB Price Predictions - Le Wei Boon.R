#R code for APAN 5200 best results for Kaggle Competition by Le Wei Boon

library(corrplot)
library(dplyr)
library(stats)
library(leaps)
library(glmnet)
library(caret)
library(MASS)
library(gbm)
library(readr)
library(xgboost)



#read data and construct a simple model
data = read.csv("analysisData.csv")

#read scoring/ testing data 
scoringData = read.csv("scoringData.csv")

#setting random seed for sampling
set.seed(1031)

#setting the split ratio with stratified sampling based on "price" as the response using createDataPartition()
split <- createDataPartition(data$price, p =0.8, list = F)

#forming the training and testing data subset from "data"
train_data <- data[split,]
test_data <- data[-split,]

train_data$zipcode <- as.numeric(train_data$zipcode)
test_data$zipcode <- as.numeric(test_data$zipcode)

#trying with a smaller selected subset of variables. Excluding variables like comments, descriptions of properties, etc.
#also excluding "square_feet" as there are too many entries in the training data and scoringData with NA values for this variable
train_data_truncated <- train_data[, c(1, 19, 28, 29, 32, 38:45, 52:61, 64:69, 72:78, 84, 91, 47)]

#remove rows with NA values
train_data_cleaned <- na.omit(train_data_truncated)


#forming the correlation matrix using cor(). I have removed all non-numeric variables from the data frame so that correlation can be computed
correlationMatrix <- cor(train_data_cleaned[, c(-(1:4),-6,-7,-12, - 13, -37)])

#extracting out the correlation coefficients for variables against "price"
correlation_with_price <- correlationMatrix[,"price"]
correlation_with_price

#creating the correlation matrix viz plot with "square"
corrplot(correlationMatrix,method = 'square',type = 'lower',diag = F)

#alternative correlation matrix viz plot with "values"
corrplot(correlationMatrix,method = 'number',type = 'lower',diag = F)


#further cutting down the num of predictors after initial correlation analysis
#removing excessive variables 
train_data_cleaned_2 <- train_data_cleaned [, c(-2, -(9:11), -(16:21), -23, -24, -(29:32), -34)]
train_data_cleaned_2


#Initial Features Selection

#Model 1: Forward subset selection 

train_data_cleaned_2 <- train_data_cleaned [, c(-2, -(11:13), -(18:23))]

#generating another subset from training data with "price" included
train_data_cleaned_predictors_price <- train_data_cleaned_2[,c(2:29)]

#for forward selection
start_mod <- lm(price ~ 1, data = train_data_cleaned_predictors_price)
empty_mod <- lm(price ~ 1, data = train_data_cleaned_predictors_price)
full_mod <- lm(price ~ ., data = train_data_cleaned_predictors_price)

forwardstepwise <- step(start_mod, 
                        scope = list(upper = full_mod, lower = empty_mod), 
                        direction = "forward")
summary(forwardstepwise)
forwardstepwise


#Model 2: Hybrid subset selection
train_data_cleaned_2 <- train_data_cleaned [, c(-2, -(11:13), -(18:23))]

#generating another subset from training data with "price" included
train_data_cleaned_predictors_price <- train_data_cleaned_2[,c(2:29)]


#for hybrid selection, 
start_mod <- lm(price ~ 1, data = train_data_cleaned_predictors_price)
empty_mod <- lm(price ~ 1, data = train_data_cleaned_predictors_price)
full_mod <- lm(price ~ ., data = train_data_cleaned_predictors_price)

hybridstepwise <- step(start_mod, 
                        scope = list(upper = full_mod, lower = empty_mod), 
                        direction = "both")
summary(hybridstepwise)
hybridstepwise


#Method 3: Lasso

train_data_cleaned_2 <- train_data_cleaned [, c(-2, -(11:13), -(18:23))]

#generating another subset from training data with "price" included
train_data_cleaned_predictors_price <- train_data_cleaned_2[,c(2:29)]

#need to set seed as cross validation (i.e. cv) is random
set.seed(1031)

#forming the x (i.e. predictors) and y(i.e. the response). The price ~.1 is to exclude the intercept
x = model.matrix(price~.-1,data=train_data_cleaned_predictors_price)
y = train_data_cleaned_predictors_price$price

#set alpha = 1 for lasso. alpha = 0 for ridge
lasso_train <- cv.glmnet(x, y, nfolds = 10, alpha = 1)

lasso_train
coef(lasso_train)

#now train with the variables identified by lasso
#omitted property_type as test_data does not have it
lm_lasso_train <- lm(price ~ neighbourhood_group_cleansed + room_type + accommodates +
                       bathrooms + bedrooms + guests_included + extra_people + minimum_nights + availability_30 + availability_90 + availability_365 +
                       number_of_reviews + number_of_reviews_ltm + review_scores_rating + review_scores_cleanliness + review_scores_location + cancellation_policy + reviews_per_month,
                     data = train_data_cleaned_predictors_price)

summary(lm_lasso_train)

#extracting the R^2 value of this linear regression
summary(lm_lasso_train)$r.squared

#predicting prices for the test data based on the lasso regression variables
pred3 <- predict(lm_lasso_train, newdata = test_data)

rmse_ana <- sqrt(mean((pred3 - test_data$price)^2))
rmse_ana



#Method 4: Ridge

train_data_cleaned_2 <- train_data_cleaned [, c(-2, -(11:13), -(18:23))]

#generating another subset from training data with "price" included
train_data_cleaned_predictors_price <- train_data_cleaned_2[,c(2:29)]

#need to set seed as cross validation (i.e. cv) is random
set.seed(1031)

#forming the x (i.e. predictors) and y(i.e. the response). The price ~.1 is to exclude the intercept
x = model.matrix(price~.-1,data=train_data_cleaned_predictors_price)
y = train_data_cleaned_predictors_price$price

#set alpha = 1 for lasso. alpha = 0 for ridge
ridge_train <- cv.glmnet(x, y, nfolds = 10, alpha = 0)

ridge_train
coef(ridge_train)

#now train with the variables identified by lasso
#omitted property_type as test_data does not have it
lm_ridge_train <- lm(price ~ neighbourhood_group_cleansed + room_type + accommodates +
                       bathrooms + bedrooms + guests_included + extra_people + minimum_nights + availability_30 + availability_90 + availability_365 +
                       number_of_reviews + number_of_reviews_ltm + review_scores_rating + review_scores_cleanliness + review_scores_location + cancellation_policy + reviews_per_month,
                     data = train_data_cleaned_predictors_price)

summary(lm_ridge_train)

#extracting the R^2 value of this linear regression
summary(lm_ridge_train)$r.squared

#predicting prices for the test data based on the ridge regression variables
pred3 <- predict(lm_ridge_train, newdata = test_data)

rmse_ana <- sqrt(mean((pred3 - test_data$price)^2))
rmse_ana


#Method 5: Principle Components Analysis (PCA) 
train_data_cleaned_2 <- train_data_cleaned [, c(-2, -(11:13), -(18:23))]

#generating another subset from training data with "price" included
train_data_cleaned_predictors_price <- train_data_cleaned_2[,c(2:29)]

#firstly, select the variables in test data to be the same as train_data_cleaned_predictors_price
test_data_cleaned_predictors_price = test_data[, c(28, 29, 32, 38:43, 52:55, 64:67, 68:78, 84, 91, 47)]

#performing PCA to transform the original predictors into new components (i.e. PC1, PC2,...)
#the threshold is set to 0.9. Meaning that we will select the new components PC1, PC2.... up to the component that gives us 90% of total original data variance.

trainPredictors = train_data_cleaned_predictors_price[, -28]
testPredictors = test_data_cleaned_predictors_price[, -28]

x = preProcess(x = trainPredictors,method = 'pca',thresh = 0.9)
trainComponents = predict(x,newdata=trainPredictors)

#remove property_type from trainComponents
trainComponents = trainComponents[, -3]
#add in price to trainComponents
trainComponents$price = train_data_cleaned_predictors_price$price

#using summary() to show the values of the different components. As shown, there are 7 new components (i.e. PC1 to PC7)
summary(trainComponents)

#printing out x will show "PCA needed 13 components to capture 90 percent of the variance"
x

#Finding R^2 value for a linear regression using the components found via PCA in qns 8.
model_train_pca <- lm(price ~., data = trainComponents)
summary(model_train_pca)

#extracting the R^2 value
summary(model_train_pca)$r.squared








#Final code section

#building an initial model to predict prices of rows in scoringData with property type that are not in the current list of training data
model3 <- lm(price ~ neighbourhood_group_cleansed + room_type + accommodates +
               bathrooms + bedrooms + guests_included + extra_people + minimum_nights + availability_30 + availability_365 +
               number_of_reviews_ltm + review_scores_rating + review_scores_location + reviews_per_month,
             data = data)

summary(model3)


#create a new small dataset comprising 1 x "Castle" and 1 x "Dome House" observations from the scoringData
#Training it with model3. Then add the price col in. Before merging these 2 observations (rows)
#back to the original "data" (i.e. analysisData). 
#Intent is to be able to use the "property_type" as a predictor in a new model. 

#creating a small dataframe by filtering rows that belong to these two property_types for the original scoringData
castle_dome_house <- scoringData %>%
  filter (property_type == "Castle" | property_type == "Dome house")

#using the earlier prediction model3 to predict the prices for these homes
pred_castle_dome_house <- predict(model3, newdata = castle_dome_house)
pred_castle_dome_house

#adding the predicted prices for these two property types back to the small dataframe
castle_dome_house$price = pred_castle_dome_house
castle_dome_house

#rearranging the columns of this very small dataframe to align with "data" (i.e. training data)
castle_dome_house_rearranged <- castle_dome_house[, c(1:46, 91, 47:90)]
castle_dome_house_rearranged

#coercing "zipcode" in the very small subset to char to align with "data" before binding by rows
castle_dome_house_rearranged$zipcode <- as.character(castle_dome_house_rearranged$zipcode)

#adding this very small dataframe to the original "data" to create a new dataset called "data_added_property_type"
data_added_property_type <- bind_rows(data, castle_dome_house_rearranged)
data_added_property_type



#Now with the training dataset pre-processed, we will explore the price distribution

#Part 1: plotting the distribution of prices based on room_type
ggplot(data_added_property_type, aes(x = room_type, y = price)) +
  geom_boxplot()

#Part 2: plotting the distribution of prices based on neighbourhood_group_cleansed
ggplot(data_added_property_type, aes(x = neighbourhood_group_cleansed, y = price)) +
  geom_boxplot()

#Plot 3: plotting the distribution of prices based on neighbourhood_group_cleansed and faceted by room_type
ggplot(data_added_property_type, aes(x = neighbourhood_group_cleansed, y = price)) +
  geom_boxplot()+
  facet_grid(vars(room_type))

#Plot 4: plotting a bar chart grouped by room_type and filled by neighbourhood_grouP-cleansed
ggplot(data=data_added_property_type,aes(y=price,x=room_type,fill=factor(neighbourhood_group_cleansed)))+ 
  geom_bar(stat="summary",fun="mean",position="dodge")



#pre-processing of the analysisData

#with standardization of numeric variables in the "data_added_property_type" and "scoringData"
scaled_data_added_property_type  <- data_added_property_type %>%
  mutate_at(c("accommodates", "bathrooms", "bedrooms", "guests_included", "extra_people", "minimum_nights", "maximum_minimum_nights", "minimum_maximum_nights", "maximum_maximum_nights", "maximum_nights_avg_ntm",
              "availability_30", "availability_365", "number_of_reviews_ltm", "review_scores_rating", "review_scores_checkin", "review_scores_communication", "review_scores_location", "reviews_per_month", "review_scores_cleanliness",
              "review_scores_value", "calculated_host_listings_count", "calculated_host_listings_count_entire_homes",
              "host_listings_count", "calculated_host_listings_count_private_rooms", "calculated_host_listings_count_shared_rooms", "review_scores_accuracy","cleaning_fee", "security_deposit"),
            ~(scale(.) %>% as.vector))
scaled_data_added_property_type

scaled_data_added_property_type$cleaning_fee[is.na(scaled_data_added_property_type$cleaning_fee)] <- 0
scaled_data_added_property_type$security_deposit[is.na(scaled_data_added_property_type$security_deposit)] <- 0


scaled_data_added_property_type$neighbourhood_group_cleansed <- as.factor(scaled_data_added_property_type$neighbourhood_group_cleansed)
scaled_data_added_property_type$neighbourhood_cleansed <- as.factor(scaled_data_added_property_type$neighbourhood_cleansed)
scaled_data_added_property_type$cancellation_policy <- as.factor(scaled_data_added_property_type$cancellation_policy)
scaled_data_added_property_type$property_type <- as.factor(scaled_data_added_property_type$property_type)
scaled_data_added_property_type$zipcode <- as.factor(scaled_data_added_property_type$zipcode)


#identifying keywords under the "name" column that correspond to more expensive properties
#followed by creating a new column "ex" to tag these more expensive properties

words <- c("luxury","Luxury", "Penthouse", "LUXURY", "LUX", "Townhome", "Beekman Tower",
           "Presidential", "Upscale", "PRESIDENTIAL","Luxurious", "SoHo", "soho", "Soho",
           "Townhouse", "townhouse")

scaled_data_added_property_type2 <- scaled_data_added_property_type
ex <- scaled_data_added_property_type2[rowSums(sapply(words, grepl, scaled_data_added_property_type2$name)) > 0, , drop = FALSE]
ex

index = as.list(ex$id)

scaled_data_added_property_type2_ex<- scaled_data_added_property_type2[scaled_data_added_property_type2$id %in% index, ]
scaled_data_added_property_type2_not_ex<- scaled_data_added_property_type2[!scaled_data_added_property_type2$id %in% index, ]

scaled_data_added_property_type2_ex$ex = 1
scaled_data_added_property_type2_not_ex$ex = 0

scaled_data_added_property_type2 <- rbind(scaled_data_added_property_type2_ex,
                                          scaled_data_added_property_type2_not_ex)

scaled_data_added_property_type2 <- scaled_data_added_property_type2[order(scaled_data_added_property_type2$id, decreasing = FALSE), ]

scaled_data_added_property_type <- scaled_data_added_property_type2


scaled_data_added_property_type3 <- scaled_data_added_property_type2_ex %>%
  select(name, zipcode, neighbourhood_group_cleansed, room_type, price)



#stratifying the original training data based on "room_type"

scaled_data_added_property_type_entirehome <- scaled_data_added_property_type %>%
  filter(room_type == "Entire home/apt") %>%
  filter(price > 0)

#splitting entirehomes into Manhattan 
scaled_data_added_property_type_entirehome_man <- scaled_data_added_property_type_entirehome %>%
  filter(neighbourhood_group_cleansed == "Manhattan")

#splitting entirehomes into Brooklyn
scaled_data_added_property_type_entirehome_brook <- scaled_data_added_property_type_entirehome %>%
  filter(neighbourhood_group_cleansed == "Brooklyn")

#splitting entirehomes into Queens
scaled_data_added_property_type_entirehome_queens <- scaled_data_added_property_type_entirehome %>%
  filter(neighbourhood_group_cleansed =="Queens")

#splitting the rest including Staten Island, Bronx 
scaled_data_added_property_type_entirehome_rest <- scaled_data_added_property_type_entirehome %>%
  filter(neighbourhood_group_cleansed == "Staten Island" | neighbourhood_group_cleansed == "Bronx")

#splitting out Hotel room
scaled_data_added_property_type_hotel <- scaled_data_added_property_type %>%
  filter(room_type == "Hotel room")

#splitting out private rooms
scaled_data_added_property_type_private <- scaled_data_added_property_type %>%
  filter(room_type == "Private room") %>%
  filter(reviews_per_month != 'NA') %>%
  filter(price > 0)

#splitting private rooms into Manhattan 
scaled_data_added_property_type_private_man <- scaled_data_added_property_type_private %>%
  filter(neighbourhood_group_cleansed == "Manhattan")

#splitting private rooms into Brooklyn
scaled_data_added_property_type_private_brook <- scaled_data_added_property_type_private %>%
  filter(neighbourhood_group_cleansed == "Brooklyn")

#splitting the rest including Staten Island, Bronx and Queens
scaled_data_added_property_type_private_rest <- scaled_data_added_property_type_private %>%
  filter(neighbourhood_group_cleansed == "Staten Island" | neighbourhood_group_cleansed == "Bronx" | neighbourhood_group_cleansed =="Queens")

#splitting out shared rooms and removing outliers with price >- 500
scaled_data_added_property_type_shared <- scaled_data_added_property_type %>%
  filter(room_type == "Shared room") %>%
  filter(price > 0 & price < 500)



#gbm for entirehomes in manhattan
#splitting into training and testing
set.seed(1234)

split = sample.split(scaled_data_added_property_type_entirehome_man$price,SplitRatio=0.8)
scaled_data_added_property_type_entirehome_man_train = scaled_data_added_property_type_entirehome_man[split, ]
scaled_data_added_property_type_entirehome_man_test = scaled_data_added_property_type_entirehome_man[!split, ]


#training gbm for entire homes in manhatten
set.seed(1031)
modelentire_man_gbm <- gbm(price ~ ex + security_deposit + cleaning_fee + neighbourhood_cleansed + zipcode + property_type + accommodates *
                             bathrooms * bedrooms + minimum_nights_avg_ntm + guests_included * extra_people + availability_30 + availability_365 +
                             number_of_reviews + review_scores_rating + review_scores_accuracy + review_scores_cleanliness + review_scores_value + reviews_per_month + 
                             cancellation_policy + calculated_host_listings_count * calculated_host_listings_count_entire_homes,                       
                           distribution = "gaussian",
                           data = scaled_data_added_property_type_entirehome_man,
                           n.trees = 3637,
                           interaction.depth = 8,
                           shrinkage = 0.010,
                           n.minobsinnode = 25)

summary(modelentire_man_gbm)


#predicting the prices for test data based on the gbm model
pre <- predict(modelentire_man_gbm, newdata = scaled_data_added_property_type_entirehome_man_test)
rmse_entire <- sqrt(mean((pre-scaled_data_added_property_type_entirehome_man_test$price)^2))
rmse_entire


#finding R-squared value
residuals = scaled_data_added_property_type_entirehome_man_test$price - pre

scaled_data_added_property_type_entirehome_man_test_meanprice = mean(scaled_data_added_property_type_entirehome_man_test$price)

# Calculate total sum of squares
tss =  sum((scaled_data_added_property_type_entirehome_man_test$price - scaled_data_added_property_type_entirehome_man_test_meanprice)^2 )

# Calculate residual sum of squares
rss =  sum(residuals^2)

# Calculate R-squared value
rsq  =  1 - (rss/tss)
rsq



#alternative lm for entirehome manhattan
modelentirehome_man <- lm(price ~ neighbourhood_cleansed + zipcode + property_type + accommodates *
                            bathrooms * bedrooms + minimum_nights + guests_included * extra_people + availability_30 + availability_365 +
                            number_of_reviews_ltm + review_scores_rating * review_scores_location *reviews_per_month + review_scores_accuracy + review_scores_value  + review_scores_rating + 
                            cancellation_policy,
                          data = scaled_data_added_property_type_entirehome_man)

summary(modelentirehome_man)

pre1111 <- predict(modelentirehome_man, newdata = scaled_data_added_property_type_entirehome_man)
rmse_entire1111 <- sqrt(mean((pre1111-scaled_data_added_property_type_entirehome_man$price)^2))
rmse_entire1111



#alternative lm for entirehome manhattan (outliers without the same zipcode)
modelentirehome_man_out <- lm(price ~ ex + neighbourhood_cleansed + property_type + accommodates *
                                bathrooms * bedrooms  + minimum_nights + guests_included * extra_people + availability_30 + availability_365 +
                                number_of_reviews_ltm + review_scores_rating * review_scores_location * reviews_per_month + review_scores_accuracy + review_scores_value  + review_scores_rating + 
                                cancellation_policy,
                              data = scaled_data_added_property_type_entirehome_man)

summary(modelentirehome_man_out)




#gbm for entirehomes in brooklyn
#splitting into training and testing
set.seed(1234)

split = sample.split(scaled_data_added_property_type_entirehome_brook$price,SplitRatio=0.8)
scaled_data_added_property_type_entirehome_brook_train = scaled_data_added_property_type_entirehome_brook[split, ]
scaled_data_added_property_type_entirehome_brook_test = scaled_data_added_property_type_entirehome_brook[!split, ]


#training gbm for entire homes in brooklyn
set.seed(1031)
modelentire_brook_gbm <- gbm(price ~ ex + security_deposit + cleaning_fee + neighbourhood_cleansed + zipcode + property_type + accommodates *
                               bathrooms * bedrooms  + minimum_nights + guests_included * extra_people + availability_30 + availability_365 +
                               number_of_reviews_ltm + review_scores_rating + review_scores_location + review_scores_accuracy +reviews_per_month+ review_scores_value +
                               cancellation_policy + calculated_host_listings_count,
                             distribution = "gaussian",
                             data = scaled_data_added_property_type_entirehome_brook,
                             n.trees = 1083,
                             interaction.depth = 15,
                             shrinkage = 0.009,
                             n.minobsinnode = 50)


summary(modelentire_brook_gbm)


#predicting the prices based on the gbm model
pre <- predict(modelentire_brook_gbm, newdata = scaled_data_added_property_type_entirehome_brook_test)
rmse_entire <- sqrt(mean((pre-scaled_data_added_property_type_entirehome_brook_test$price)^2))
rmse_entire


#finding R-squared value
residuals = scaled_data_added_property_type_entirehome_brook_test$price - pre

scaled_data_added_property_type_entirehome_brook_test_meanprice = mean(scaled_data_added_property_type_entirehome_brook_test$price)

# Calculate total sum of squares
tss =  sum((scaled_data_added_property_type_entirehome_brook_test$price - scaled_data_added_property_type_entirehome_brook_test_meanprice)^2 )

# Calculate residual sum of squares
rss =  sum(residuals^2)

# Calculate R-squared value
rsq  =  1 - (rss/tss)
rsq





#alternative lm for entirehome in brooklyn
modelentirehome_brook <- lm(price ~ ex + neighbourhood_cleansed + zipcode + property_type + accommodates *
                              bathrooms * bedrooms  + minimum_nights + guests_included * extra_people + availability_30 + availability_365 +
                              number_of_reviews_ltm + review_scores_rating * review_scores_location* reviews_per_month + review_scores_accuracy + review_scores_value  + review_scores_rating,
                            data = scaled_data_added_property_type_entirehome_brook)

summary(modelentirehome_brook)

pre1112 <- predict(modelentirehome_brook, newdata = scaled_data_added_property_type_entirehome_brook)
rmse_entire1112 <- sqrt(mean((pre1112-scaled_data_added_property_type_entirehome_brook$price)^2))
rmse_entire1112


#alternative lm for entirehome brooklyn (outliers that cannot match zipcode)
modelentirehome_brook_out <- lm(price ~ ex + neighbourhood_cleansed + property_type + accommodates *
                                  bathrooms * bedrooms  + minimum_nights + guests_included * extra_people + availability_30 + availability_365 +
                                  number_of_reviews_ltm + review_scores_rating * review_scores_location * reviews_per_month + review_scores_accuracy + review_scores_value  + review_scores_rating,
                                data = scaled_data_added_property_type_entirehome_brook)

summary(modelentirehome_brook_out)






#gbm for entirehomes in queens
#splitting into training and testing
set.seed(1234)

split = sample.split(scaled_data_added_property_type_entirehome_queens$price,SplitRatio=0.8)
scaled_data_added_property_type_entirehome_queens_train = scaled_data_added_property_type_entirehome_queens[split, ]
scaled_data_added_property_type_entirehome_queens_test = scaled_data_added_property_type_entirehome_queens[!split, ]


#training gbm for entire homes in queens
set.seed(1031)
modelentire_queens_gbm <- gbm(price ~ ex + host_listings_count + security_deposit + cleaning_fee + neighbourhood_cleansed + zipcode + property_type + accommodates *
                                bathrooms * bedrooms  + minimum_nights_avg_ntm + guests_included * extra_people + availability_30 + availability_365 +
                                number_of_reviews + review_scores_rating * review_scores_location* reviews_per_month + review_scores_accuracy + review_scores_value  + review_scores_rating +
                                cancellation_policy + calculated_host_listings_count + calculated_host_listings_count_entire_homes,
                              distribution = "gaussian",
                              data = scaled_data_added_property_type_entirehome_queens,
                              n.trees = 656,
                              interaction.depth = 3,
                              shrinkage = 0.01,
                              n.minobsinnode = 11)

summary(modelentire_queens_gbm)


#predicting the prices based on the gbm model
pre <- predict(modelentire_queens_gbm, newdata = scaled_data_added_property_type_entirehome_queens_test)
rmse_entire <- sqrt(mean((pre-scaled_data_added_property_type_entirehome_queens_test$price)^2))
rmse_entire

#finding R-squared value
residuals = scaled_data_added_property_type_entirehome_queens_test$price - pre

scaled_data_added_property_type_entirehome_queens_test_meanprice = mean(scaled_data_added_property_type_entirehome_queens_test$price)

# Calculate total sum of squares
tss =  sum((scaled_data_added_property_type_entirehome_queens_test$price - scaled_data_added_property_type_entirehome_queens_test_meanprice)^2 )

# Calculate residual sum of squares
rss =  sum(residuals^2)

# Calculate R-squared value
rsq  =  1 - (rss/tss)
rsq




#gbm for entirehomes in Staten Island and Bronx
#splitting into training and testing
set.seed(1234)

split = sample.split(scaled_data_added_property_type_entirehome_rest$price,SplitRatio=0.8)
scaled_data_added_property_type_entirehome_rest_train = scaled_data_added_property_type_entirehome_rest[split, ]
scaled_data_added_property_type_entirehome_rest_test = scaled_data_added_property_type_entirehome_rest[!split, ]


#training gbm for entire homes in Staten Island and Bronx
set.seed(1031)
modelentire_rest_gbm <- gbm(price ~ ex + neighbourhood_cleansed + zipcode + property_type + accommodates *
                              bathrooms * bedrooms  + minimum_nights + guests_included * extra_people + availability_30 + availability_365 +
                              number_of_reviews + review_scores_rating * review_scores_location* reviews_per_month + review_scores_accuracy + review_scores_value  + review_scores_rating +
                              cancellation_policy + calculated_host_listings_count * calculated_host_listings_count_entire_homes,
                            distribution = "gaussian",
                            data = scaled_data_added_property_type_entirehome_rest,
                            n.trees = 950,
                            interaction.depth = 8,
                            shrinkage = 0.01,
                            n.minobsinnode = 5)

summary(modelentire_rest_gbm)


#predicting the prices based on the gbm model
pre <- predict(modelentire_rest_gbm, newdata = scaled_data_added_property_type_entirehome_rest_test)
rmse_entire <- sqrt(mean((pre-scaled_data_added_property_type_entirehome_rest_test$price)^2))
rmse_entire


#finding R-squared value
residuals = scaled_data_added_property_type_entirehome_rest_test$price - pre

scaled_data_added_property_type_entirehome_rest_test_meanprice = mean(scaled_data_added_property_type_entirehome_rest_test$price)

# Calculate total sum of squares
tss =  sum((scaled_data_added_property_type_entirehome_rest_test$price - scaled_data_added_property_type_entirehome_rest_test_meanprice)^2 )

# Calculate residual sum of squares
rss =  sum(residuals^2)

# Calculate R-squared value
rsq  =  1 - (rss/tss)
rsq






#alternative lm for entirehome staten island, bronx, queens
modelentirehome_rest <- lm(price ~ ex + neighbourhood_cleansed + zipcode + property_type + accommodates *
                             bathrooms * bedrooms + minimum_nights + availability_30 +
                             review_scores_rating * review_scores_location * reviews_per_month + review_scores_rating,
                           data = scaled_data_added_property_type_entirehome_rest)

summary(modelentirehome_rest)

pre1113 <- predict(modelentirehome_rest, newdata = scaled_data_added_property_type_entirehome_rest)
rmse_entire1113 <- sqrt(mean((pre1113-scaled_data_added_property_type_entirehome_rest$price)^2))
rmse_entire1113

#using lm for entirehome staten island, bronx, queens (outliers that cannot match zipcode)
modelentirehome_rest_out <- lm(price ~ ex + accommodates *
                                 bathrooms * bedrooms + minimum_nights + availability_30 +
                                 review_scores_rating * review_scores_location * reviews_per_month + review_scores_rating,
                               data = scaled_data_added_property_type_entirehome_rest)

summary(modelentirehome_rest_out)





#gbm for private rooms in manhattan
#splitting into training and testing
set.seed(1234)

split = sample.split(scaled_data_added_property_type_private_man$price,SplitRatio=0.8)
scaled_data_added_property_type_private_man_train = scaled_data_added_property_type_private_man[split, ]
scaled_data_added_property_type_private_man_test = scaled_data_added_property_type_private_man[!split, ]


#training gbm for private in manhattan
set.seed(1031)
modelprivate_man_gbm <- gbm(price ~ ex + cleaning_fee + neighbourhood_cleansed + zipcode + property_type + accommodates *
                              bathrooms * bedrooms + guests_included * extra_people + minimum_nights + availability_30 + availability_365 +
                              number_of_reviews_ltm + review_scores_rating * reviews_per_month + review_scores_value + calculated_host_listings_count * calculated_host_listings_count_private_rooms,
                            distribution = "gaussian",
                            data = scaled_data_added_property_type_private_man,
                            n.trees = 1071,
                            interaction.depth = 7,
                            shrinkage = 0.01,
                            n.minobsinnode = 30)

summary(modelprivate_man_gbm)


#predicting the prices for test data based on the gbm model
pre <- predict(modelprivate_man_gbm, newdata = scaled_data_added_property_type_private_man_test)
rmse_entire <- sqrt(mean((pre-scaled_data_added_property_type_private_man_test$price)^2))
rmse_entire


#finding R-squared value
residuals = scaled_data_added_property_type_private_man_test$price - pre

scaled_data_added_property_type_private_man_test_meanprice = mean(scaled_data_added_property_type_private_man_test$price)

# Calculate total sum of squares
tss =  sum((scaled_data_added_property_type_private_man_test$price - scaled_data_added_property_type_private_man_test_meanprice)^2 )

# Calculate residual sum of squares
rss =  sum(residuals^2)

# Calculate R-squared value
rsq  =  1 - (rss/tss)
rsq




#gbm for private in brooklyn
#splitting into training and testing
set.seed(1234)

split = sample.split(scaled_data_added_property_type_private_brook$price,SplitRatio=0.8)
scaled_data_added_property_type_private_brook_train = scaled_data_added_property_type_private_brook[split, ]
scaled_data_added_property_type_private_brook_test = scaled_data_added_property_type_private_brook[!split, ]


#training gbm for private in brooklyn
set.seed(1031)
modelprivate_brook_gbm <- gbm(price ~ ex + cleaning_fee + neighbourhood_cleansed + zipcode + property_type + accommodates *
                                bathrooms * bedrooms + guests_included * extra_people + minimum_nights + availability_30 + availability_365 +
                                number_of_reviews_ltm + review_scores_rating * review_scores_location * reviews_per_month + review_scores_value + calculated_host_listings_count + calculated_host_listings_count_private_rooms,
                              distribution = "gaussian",
                              data = scaled_data_added_property_type_private_brook,
                              n.trees = 850,
                              interaction.depth = 2,
                              shrinkage = 0.01,
                              n.minobsinnode = 50)

summary(modelprivate_brook_gbm)


#predicting the prices based on the gbm model
pre <- predict(modelprivate_brook_gbm, newdata = scaled_data_added_property_type_private_brook_test)
rmse_entire <- sqrt(mean((pre-scaled_data_added_property_type_private_brook_test$price)^2))
rmse_entire


#finding R-squared value
residuals = scaled_data_added_property_type_private_brook_test$price - pre

scaled_data_added_property_type_private_brook_test_meanprice = mean(scaled_data_added_property_type_private_brook_test$price)

# Calculate total sum of squares
tss =  sum((scaled_data_added_property_type_private_brook_test$price - scaled_data_added_property_type_private_brook_test_meanprice)^2 )

# Calculate residual sum of squares
rss =  sum(residuals^2)

# Calculate R-squared value
rsq  =  1 - (rss/tss)
rsq



#gbm for private in rest of boroughs
#splitting into training and testing
set.seed(1234)

split = sample.split(scaled_data_added_property_type_private_rest$price,SplitRatio=0.8)
scaled_data_added_property_type_private_rest_train = scaled_data_added_property_type_private_rest[split, ]
scaled_data_added_property_type_private_rest_test = scaled_data_added_property_type_private_rest[!split, ]


#training gbm for private in rest
set.seed(1031)
modelprivate_rest_gbm <- gbm(price ~ ex + cleaning_fee + neighbourhood_cleansed + zipcode + neighbourhood_group_cleansed + property_type + accommodates *
                               bathrooms * bedrooms + guests_included * extra_people + minimum_nights + availability_30 + availability_365 +
                               number_of_reviews + review_scores_rating * review_scores_location * reviews_per_month + calculated_host_listings_count * calculated_host_listings_count_private_rooms,
                             distribution = "gaussian",
                             data = scaled_data_added_property_type_private_rest,
                             n.trees = 1050,
                             interaction.depth = 2,
                             shrinkage = 0.01,
                             n.minobsinnode = 130)

summary(modelprivate_rest_gbm)


#predicting the prices based on the gbm model
pre <- predict(modelprivate_rest_gbm, newdata = scaled_data_added_property_type_private_rest_test)
rmse_entire <- sqrt(mean((pre-scaled_data_added_property_type_private_rest_test$price)^2))
rmse_entire



#finding R-squared value
residuals = scaled_data_added_property_type_private_rest_test$price - pre

scaled_data_added_property_type_private_rest_test_meanprice = mean(scaled_data_added_property_type_private_rest_test$price)

# Calculate total sum of squares
tss =  sum((scaled_data_added_property_type_private_rest_test$price - scaled_data_added_property_type_private_rest_test_meanprice)^2 )

# Calculate residual sum of squares
rss =  sum(residuals^2)

# Calculate R-squared value
rsq  =  1 - (rss/tss)
rsq




#alternative lm model for private
modelprivate <- lm(price ~ ex + neighbourhood_cleansed + zipcode + neighbourhood_group_cleansed + property_type + accommodates *
                     bathrooms * bedrooms + guests_included * extra_people + minimum_nights + availability_30 + availability_365 +
                     number_of_reviews_ltm + review_scores_rating * review_scores_location * reviews_per_month + review_scores_value + calculated_host_listings_count * calculated_host_listings_count_private_rooms,
                   data = scaled_data_added_property_type_private)

summary(modelprivate)

pre777 <- predict(modelprivate, newdata = scaled_data_added_property_type_private)
rmse_entire777 <- sqrt(mean((pre777-scaled_data_added_property_type_private$price)^2))
rmse_entire777





#alternative lm model for private (outliers that cannot match zipcode)
modelprivate_out <- lm(price ~ ex + neighbourhood_group_cleansed + property_type + accommodates *
                         bathrooms * bedrooms + guests_included * extra_people + minimum_nights + availability_30 + availability_365 +
                         number_of_reviews_ltm + review_scores_rating * review_scores_location * reviews_per_month + review_scores_value + calculated_host_listings_count * calculated_host_listings_count_private_rooms,
                       data = scaled_data_added_property_type_private)

summary(modelprivate_out)



#gbm for shared rooms
#splitting into training and testing
set.seed(1234)

split = sample.split(scaled_data_added_property_type_shared$price,SplitRatio=0.8)
scaled_data_added_property_type_shared_train = scaled_data_added_property_type_shared[split, ]
scaled_data_added_property_type_shared_test = scaled_data_added_property_type_shared[!split, ]


#training gbm for shared
set.seed(1031)
modelshared_gbm <- gbm(price ~ cleaning_fee + neighbourhood_cleansed + zipcode + neighbourhood_group_cleansed + property_type + accommodates *
                         bathrooms + guests_included * extra_people + minimum_nights_avg_ntm + availability_30 + availability_365 +
                         number_of_reviews_ltm + review_scores_rating * review_scores_location * reviews_per_month + review_scores_value + calculated_host_listings_count_shared_rooms,
                       distribution = "gaussian",
                       data = scaled_data_added_property_type_shared,
                       n.trees = 400,
                       interaction.depth = 3,
                       shrinkage = 0.01,
                       n.minobsinnode = 11)

summary(modelshared_gbm)


#predicting the prices based on the gbm model
pre <- predict(modelshared_gbm, newdata = scaled_data_added_property_type_shared_test)
rmse_entire <- sqrt(mean((pre-scaled_data_added_property_type_shared_test$price)^2))
rmse_entire

#finding R-squared value
residuals = scaled_data_added_property_type_shared_test$price - pre

scaled_data_added_property_type_shared_test_meanprice = mean(scaled_data_added_property_type_shared_test$price)

# Calculate total sum of squares
tss =  sum((scaled_data_added_property_type_shared_test$price - scaled_data_added_property_type_shared_test_meanprice)^2 )

# Calculate residual sum of squares
rss =  sum(residuals^2)

# Calculate R-squared value
rsq  =  1 - (rss/tss)
rsq





#alternative lm for shared
modelshared <- lm(price ~ ex + neighbourhood_cleansed + zipcode + neighbourhood_group_cleansed + property_type + accommodates *
                    bathrooms + guests_included * extra_people + minimum_nights + availability_30 + availability_365 +
                    number_of_reviews_ltm + review_scores_rating * review_scores_location * reviews_per_month + review_scores_value + calculated_host_listings_count_shared_rooms,
                  data = scaled_data_added_property_type_shared)

summary(modelshared)

pre888 <- predict(modelshared, newdata = scaled_data_added_property_type_shared)
rmse_entire888 <- sqrt(mean((pre888-scaled_data_added_property_type_shared$price)^2))
rmse_entire888

#alternative lm for shared (outliers that cannot match zipcode)
modelshared_out <- lm(price ~ ex + neighbourhood_group_cleansed + property_type + accommodates *
                        bathrooms + guests_included * extra_people + minimum_nights + availability_30 + availability_365 +
                        number_of_reviews_ltm + review_scores_rating * review_scores_location * reviews_per_month + review_scores_value + calculated_host_listings_count_shared_rooms,
                      data = scaled_data_added_property_type_shared)

summary(modelshared_out)




#pre-processing of testing data to predict prices

#standardizing the numerical variables for testing data
scaled_scoringData  <- scoringData %>%
  mutate_at(c("accommodates", "bathrooms", "bedrooms", "guests_included", "extra_people", "minimum_nights", "maximum_minimum_nights", "minimum_maximum_nights", "maximum_maximum_nights", "maximum_nights_avg_ntm",
              "availability_30", "availability_365", "number_of_reviews_ltm", "review_scores_rating", "review_scores_checkin", "review_scores_communication", "review_scores_location", "reviews_per_month", "review_scores_cleanliness",
              "review_scores_value", "calculated_host_listings_count", "calculated_host_listings_count_entire_homes",
              "host_listings_count", "calculated_host_listings_count_private_rooms", "calculated_host_listings_count_shared_rooms", "review_scores_accuracy", "cleaning_fee", "security_deposit"),
            ~(scale(.) %>% as.vector))
scaled_scoringData

#replacing all NA values with 0
scaled_scoringData$cleaning_fee[is.na(scaled_scoringData$cleaning_fee)] <- 0
scaled_scoringData$security_deposit[is.na(scaled_scoringData$security_deposit)] <- 0



#converting to factors for the gbm models
scaled_scoringData$zipcode <- as.factor(scaled_scoringData$zipcode)
scaled_scoringData$neighbourhood_group_cleansed <- as.factor(scaled_scoringData$neighbourhood_group_cleansed)
scaled_scoringData$neighbourhood_cleansed <- as.factor(scaled_scoringData$neighbourhood_cleansed)
scaled_scoringData$cancellation_policy <- as.factor(scaled_scoringData$cancellation_policy)
scaled_scoringData$property_type <- as.factor(scaled_scoringData$property_type)


#identifying keywords in testing data to create new variable "ex" for more ex properties
words <- c("luxury","Luxury", "Penthouse", "LUXURY", "LUX", "Townhome", "Beekman Tower",
           "Presidential", "Upscale", "PRESIDENTIAL","Luxurious", "SoHo", "soho", "Soho",
           "Townhouse", "townhouse")
scaled_scoringData2 <- scaled_scoringData
ex <- scaled_scoringData2[rowSums(sapply(words, grepl, scaled_scoringData2$name)) > 0, , drop = FALSE]
ex

index = as.list(ex$id)

scaled_scoringData2_ex<- scaled_scoringData2[scaled_scoringData2$id %in% index, ]
scaled_scoringData2_not_ex<- scaled_scoringData2[!scaled_scoringData2$id %in% index, ]

scaled_scoringData2_ex$ex = 1
scaled_scoringData2_not_ex$ex = 0

scaled_scoringData2 <- rbind(scaled_scoringData2_ex,
                             scaled_scoringData2_not_ex)

scaled_scoringData2 <- scaled_scoringData2[order(scaled_scoringData2$id, decreasing = FALSE), ]

scaled_scoringData <- scaled_scoringData2



#filtering out entire homes
scaled_scoringData_entirehome <- scaled_scoringData %>%
  filter(room_type == "Entire home/apt")

#splitting entirehomes into Manhattan 
scaled_scoringData_entirehome_man <- scaled_scoringData_entirehome %>%
  filter(neighbourhood_group_cleansed == "Manhattan")


#splitting entirehomes into Brooklyn
scaled_scoringData_entirehome_brook <- scaled_scoringData_entirehome %>%
  filter(neighbourhood_group_cleansed == "Brooklyn") 


#splitting entirehomes into queens
scaled_scoringData_entirehome_queens <- scaled_scoringData_entirehome %>%
  filter(neighbourhood_group_cleansed == "Queens") 


#splitting the rest including Staten Island, Bronx 
scaled_scoringData_entirehome_rest <- scaled_scoringData_entirehome %>%
  filter(neighbourhood_group_cleansed == "Staten Island" | neighbourhood_group_cleansed == "Bronx")


#splitting out Hotel room
scaled_scoringData_hotel <- scaled_scoringData %>%
  filter(room_type == "Hotel room")


#splitting out private rooms
scaled_scoringData_private <- scaled_scoringData %>%
  filter(room_type == "Private room")

#splitting private into Manhattan 
scaled_scoringData_private_man <- scaled_scoringData_private %>%
  filter(neighbourhood_group_cleansed == "Manhattan")


#splitting private into Brooklyn
scaled_scoringData_private_brook <- scaled_scoringData_private %>%
  filter(neighbourhood_group_cleansed == "Brooklyn") 


#splitting the rest including Staten Island, Bronx and Queens
scaled_scoringData_private_rest <- scaled_scoringData_private %>%
  filter(neighbourhood_group_cleansed == "Staten Island" | neighbourhood_group_cleansed == "Bronx" | neighbourhood_group_cleansed =="Queens")


#splitting out shared rooms
scaled_scoringData_shared <- scaled_scoringData %>%
  filter(room_type == "Shared room")


#predicting prices for properties based on their respective models
pred_entirehome_man <- predict(modelentire_man_gbm, newdata = scaled_scoringData_entirehome_man)

pred_entirehome_brook <- predict(modelentire_brook_gbm, newdata = scaled_scoringData_entirehome_brook)

pred_entirehome_queens <- predict(modelentire_queens_gbm, newdata = scaled_scoringData_entirehome_queens)

pred_entirehome_rest <- predict(modelentire_rest_gbm, newdata = scaled_scoringData_entirehome_rest)

pred_hotel <- predict(modelprivate_man_gbm, newdata = scaled_scoringData_hotel)

pred_private_man <- predict(modelprivate_man_gbm, newdata = scaled_scoringData_private_man)

pred_private_brook <- predict(modelprivate_brook_gbm, newdata = scaled_scoringData_private_brook)

pred_private_rest <- predict(modelprivate_rest_gbm, newdata = scaled_scoringData_private_rest)

pred_shared <- predict(modelshared_gbm, newdata = scaled_scoringData_shared)




#creating a new column for testing data with the predicted prices
scaled_scoringData_entirehome_man$price <- pred_entirehome_man

scaled_scoringData_entirehome_brook$price <- pred_entirehome_brook

scaled_scoringData_entirehome_queens$price <- pred_entirehome_queens

scaled_scoringData_entirehome_rest$price <- pred_entirehome_rest

scaled_scoringData_hotel$price <- pred_hotel

scaled_scoringData_private_man$price <- pred_private_man

scaled_scoringData_private_brook$price <- pred_private_brook

scaled_scoringData_private_rest$price <- pred_private_rest

scaled_scoringData_shared$price <- pred_shared



scaled_scoringData_entirehome_man_id_price <- scaled_scoringData_entirehome_man [, c(1,92)]

scaled_scoringData_entirehome_brook_id_price <- scaled_scoringData_entirehome_brook [, c(1,92)]

scaled_scoringData_entirehome_queens_id_price <- scaled_scoringData_entirehome_queens [, c(1,92)]

scaled_scoringData_entirehome_rest_id_price <- scaled_scoringData_entirehome_rest [, c(1,92)]

scaled_scoringData_hotel_id_price <- scaled_scoringData_hotel [, c(1,92)]

scaled_scoringData_private_man_id_price <- scaled_scoringData_private_man [, c(1,92)]

scaled_scoringData_private_brook_id_price <- scaled_scoringData_private_brook [, c(1,92)]

scaled_scoringData_private_rest_id_price <- scaled_scoringData_private_rest [, c(1,92)]

scaled_scoringData_shared_id_price <- scaled_scoringData_shared [, c(1,92)]


#combining by rows to obtain the entire set of testing data
scaled_scoringData_price <- rbind(scaled_scoringData_entirehome_man_id_price,
                                  scaled_scoringData_entirehome_brook_id_price,
                                  scaled_scoringData_entirehome_queens_id_price,
                                  scaled_scoringData_entirehome_rest_id_price,
                                  scaled_scoringData_hotel_id_price,
                                  scaled_scoringData_private_man_id_price,
                                  scaled_scoringData_private_brook_id_price,
                                  scaled_scoringData_private_rest_id_price,
                                  scaled_scoringData_shared_id_price)


#replacing all negative predicted price and small values to default $0
scaled_scoringData_price_negative <- scaled_scoringData_price %>%
  filter(price <= 0) %>%
  mutate (price = 0)

#keeping the rest of predicted price
scaled_scoringData_price_remaining <- scaled_scoringData_price %>%
  filter(price > 0)

scaled_scoringData_price <- rbind(scaled_scoringData_price_remaining,
                                  scaled_scoringData_price_negative)

scaled_scoringData_price_ordered <- scaled_scoringData_price[order(scaled_scoringData_price$id, decreasing = FALSE), ]

#Construct submission from prediction
submissionFile = data.frame(id = scaled_scoringData_price_ordered$id, price = scaled_scoringData_price_ordered$price)
write.csv(submissionFile, 'submission_final_submission_le_wei_boon.csv', row.names = F)





