library(jpeg)
library(cluster) #ellipsoidhull, clara
library(fpc)
library(Matrix)

setwd("D:/Projects/RasterPolygonBuilder/Supplements")

#X0<-readJPEG("Image__2022-12-28__23-58-14_crop4.jpg")
#X0<-readJPEG("Image__2023-01-03__18-08-15_crop.jpg")
X0<-readJPEG("Image__2023-01-05__00-52-09-crop.jpg")

N0<-attributes(X0)$dim[1]*attributes(X0)$dim[2]
Y0<-array(dim=c(N0, attributes(X0)$dim[3]))
Y1<-array(dim=c(N0, 2))

n<-1
for(i in 1:attributes(X0)$dim[1]) {
    for(j in 1:attributes(X0)$dim[2]) {
        Y0[n,3] <- X0[i,j,1]
        Y0[n,2] <- X0[i,j,2]
        Y0[n,1] <- X0[i,j,3]
        
        Y1[n,] <- X0[i,j,1:2]
        n <- n+1
    }
}

Y0<-Y0 * 255
#Y0<-Y0[Y0[,1]<100,]
#Y0<-Y0[Y0[,2]<130,]
e0<-ellipsoidhull(Y0)

solve(t(chol(e0$cov)))
solve(e0$cov)
e0$loc





Y2<-Y1[Y1[,2]<0.9,]
Y3<-Y2[Y2[,2]>0.1,]

Y3<-Y3*255
e3<-ellipsoidhull(Y3)

ellipse_boundary <- predict.ellipsoid(e3)


windows()
plot(0,0, xlim = range(ellipse_boundary[,1], na.rm=T), ylim = range(ellipse_boundary[,2], na.rm=T), type = "n")
lines(ellipse_boundary, col="lightgreen", lwd=3)
points(Y3, col="red", pch=".")



