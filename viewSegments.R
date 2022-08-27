
library(zoom)

setwd("D:\\Projects\\RasterPolygonBuilder\\VSStatistics")

plotPolygons <- function(file_name, createNew = F, colors = NA) {
	max_col <- max(count.fields(file_name, sep = ','))
	col_names = paste0("V",seq_len(max_col))
	size_segments <- read.csv(file_name, col.names=col_names, strip.white=T, header=F, row.names = NULL)


	x <- size_segments[, seq(1, max_col, by=2)]
	y <- size_segments[, seq(2, max_col, by=2)]

	if(createNew ) {
		plot(0,0, xlim = range(x, na.rm=T), ylim = range(y, na.rm=T), type = "n")
	}

	for(j in 1:nrow(size_segments)) { 
		do_expand <- FALSE
		if(is.na(x[j,length(x[j,])])) {
			x[j,length(x[j,])] = x[j, 1]
			y[j,length(x[j,])] = y[j, 1]
		} 
		else {
			do_expand <- TRUE
		} 
	
		a <- as.numeric(na.omit(as.numeric(x[j,])))
		b <- as.numeric(na.omit(as.numeric(y[j,])))
	
		if(do_expand) {
			a <- as.numeric(cbind(t(a), c(x[j, 1]), deparse.level = 0))
			b <- as.numeric(cbind(t(b), c(y[j, 1]), deparse.level = 0))
		}

		if(is.na(colors)) {
			colors = sample(rainbow(nrow(size_segments)));
		}
	
		lines(a,b, col=colors)
	}
} 

file_name <- "2022Jul24_133417_402_out.csv"
file_name <- "2022Jul25_210130_485_LbBead_UnloadFailed.csv"
file_name <- "2022Jul25_231839_249_LbBead_UnloadFailed.csv"
file_name <- "2022Jul25_232609_770_LbBead_UnloadFailed.csv"
file_name <- "2022Jul25_233310_353_LbBead_UnloadFailed.csv"
file_name <- "2022Jul26_00314_313_LbBead_UnloadFailed.csv"
file_name <- "2022Jul26_90609_960_LbBead_UnloadFailed.csv"
file_name <- "2022Jul29_212552_057_out.csv"
file_name <- "2022Jul29_214104_298_LoadFragment.csv"

plotPolygons("2022Aug26_185105_397_ClassifyBits0.csv", T, "blue")
plotPolygons("2022Aug26_185105_398_ClassifyBits1.csv", F, "red")
plotPolygons("2022Aug26_185105_399_ClassifyBits2.csv", F, "brown")
plotPolygons("2022Aug26_185105_399_ClassifyBits4.csv", F, "purple")
plotPolygons("2022Aug26_185105_400_ClassifyBits8.csv", F, "magenta")
plotPolygons("2022Aug26_185105_398_ClassifyBits18.csv", F, "aquamarine")
plotPolygons("2022Aug26_185105_399_ClassifyBits24.csv", F, "black")






#x1 <- c(-979, 2603, 1631, 1631, -979) 
#y1 <- c(46213, 48601, 46876, 46876, 46213)
#lines(x1,y1, col=sample(rainbow(nrow(size_segments))))

#x1 <- c(23552, 23552, 24576, 24576, 23552)
#y1 <- c(52224, 53460, 53036, 53248, 52224)
#lines(x1,y1, col="red")


file_name <- "2022Jul29_214121_714_LoadFragment.csv"

max_col <- max(count.fields(file_name, sep = ','))
col_names = paste0("V",seq_len(max_col))
size_segments <- read.csv(file_name, col.names=col_names, strip.white=T, header=F, row.names = NULL)


x <- size_segments[, seq(1, max_col, by=2)]
y <- size_segments[, seq(2, max_col, by=2)]

for(j in 1:nrow(size_segments)) { 
	do_expand <- FALSE
	if(is.na(x[j,length(x[j,])])) {
		x[j,length(x[j,])] = x[j, 1]
		y[j,length(x[j,])] = y[j, 1]
	} 
	else {
		do_expand <- TRUE
	} 
	
	a <- as.numeric(na.omit(as.numeric(x[j,])))
	b <- as.numeric(na.omit(as.numeric(y[j,])))
	
	if(do_expand) {
		a <- as.numeric(cbind(t(a), c(x[j, 1]), deparse.level = 0))
		b <- as.numeric(cbind(t(b), c(y[j, 1]), deparse.level = 0))
	}
	
	lines(a,b, col=sample(rainbow(nrow(size_segments))))
}


file_name <- "2022Jul29_220557_858_contour_2out.csv"

max_col <- max(count.fields(file_name, sep = ','))
col_names = paste0("V",seq_len(max_col))
size_segments <- read.csv(file_name, col.names=col_names, strip.white=T, header=F, row.names = NULL)


x <- size_segments[, seq(1, max_col, by=2)]
y <- size_segments[, seq(2, max_col, by=2)]

for(j in 1:nrow(size_segments)) { 
	do_expand <- FALSE
	if(is.na(x[j,length(x[j,])])) {
		x[j,length(x[j,])] = x[j, 1]
		y[j,length(x[j,])] = y[j, 1]
	} 
	else {
		do_expand <- TRUE
	} 
	
	a <- as.numeric(na.omit(as.numeric(x[j,])))
	b <- as.numeric(na.omit(as.numeric(y[j,])))
	
	if(do_expand) {
		a <- as.numeric(cbind(t(a), c(x[j, 1]), deparse.level = 0))
		b <- as.numeric(cbind(t(b), c(y[j, 1]), deparse.level = 0))
	}
	
	lines(a,b, col="blue")
}


zm()







file_name <- "2020Oct2_151402_008_ShapesBackScaled.csv"

max_col <- max(count.fields(file_name, sep = ','))
col_names = paste0("V",seq_len(max_col))
size_segments <- read.csv(file_name, col.names=col_names, strip.white=T, header=F, row.names = NULL)


x <- size_segments[, seq(1, max_col, by=2)]
y <- size_segments[, seq(2, max_col, by=2)]

plot(0,0, xlim = range(x, na.rm=T), ylim = range(y, na.rm=T), type = "n")

for(j in 1:nrow(size_segments)) { 
	do_expand <- FALSE
	if(is.na(x[j,length(x[j,])])) {
		x[j,length(x[j,])] = x[j, 1]
		y[j,length(x[j,])] = y[j, 1]
	} 
	else {
		do_expand <- TRUE
	} 
	
	a <- as.numeric(na.omit(as.numeric(x[j,])))
	b <- as.numeric(na.omit(as.numeric(y[j,])))
	
	if(do_expand) {
		a <- as.numeric(cbind(t(a), c(x[j, 1]), deparse.level = 0))
		b <- as.numeric(cbind(t(b), c(y[j, 1]), deparse.level = 0))
	}
	
	lines(a,b, col="blue")
}




x=23552, y=52224
x=23552, y=53460
x=24576, y=53036
x=24576, y=53248
x=23552, y=52224
fixed segment

x=24576, y=56388
x=24576, y=57744
x=25600, y=56944
x=25600, y=57276
x=24576, y=56388
fixed segment

x=52736, y=77916
x=52736, y=79322
x=51712, y=78374
x=51712, y=78756
x=52736, y=77916
fixed segment

x=45056, y=78886
x=45056, y=80346
x=46080, y=79398
x=46080, y=79834
x=45056, y=78886
fixed segment

x=52736, y=79910
x=52736, y=81108
x=51712, y=80684
x=51712, y=80858
x=52736, y=79910
fixed segment

x=34816, y=2077
x=34816, y=3284
x=35840, y=2860
x=35840, y=3043
x=34816, y=2077
fixed segment

x=14848, y=39024
x=14848, y=40448
x=15872, y=39424
x=15872, y=39824
x=14848, y=39024
fixed segment

x=14848, y=40448
x=15872, y=40448
x=15872, y=39424
x=15872, y=39424
x=14848, y=40448
fixed segment





j<-1
a <- as.numeric(na.omit(as.numeric(x[j,])))
b <- as.numeric(na.omit(as.numeric(y[j,])))
plot(0,0,xlim = range(a, na.rm=T),ylim = range(b, na.rm=T),type = "n")
lines(a,b)



x <- numeric(max_col/2)
y <- numeric(max_col/2)


plot(y ~ x, na.omit(subset[1:2]), type = "l", xlim = range(x, na.rm=T), ylim = range(y, na.rm=T))




zoomplot.zoom( locator(2) )
zoomplot.zoom(fact=2,x=0,y=0)
zoomplot.zoom(fact=2,x=0,y=0)


zoomplot.zoom(fact=2,locator(1))



size_segments <- read.csv("VSStatistics2020Sep30_202543_077.csv", strip.white=T, header=F, row.names = NULL)




