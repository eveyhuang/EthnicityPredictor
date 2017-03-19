data <- read.csv("/Users/apple/Desktop/Zach/EthnicityPredictor/BerkeleyStudents.csv")
data1<- data[sample(nrow(data),size=50000),]
nrow(data1[data1$Japanese==1,])
data1 <- rbind(data[data$Japanese==1,],data[data$Japanese==0,][sample(nrow(data[data$Japanese==0,]),size=4570),])
write.csv(data1, file = "/Users/apple/Desktop/Zach/EthnicityPredictor/5percentdata.csv")
