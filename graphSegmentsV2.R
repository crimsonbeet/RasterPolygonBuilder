library("extrafont")
font_import()
extrafont::loadfonts(device="win")
library("tidygraph")
library("igraph")
library("ggraph")

data.dir <- "D:\\Projects\\RasterPolygonBuilder\\VSStatistics\\"




index.timestamps <- read.csv(sprintf("%s%s", data.dir, "mas_index.txt"), header=FALSE)

if(nrow(index.timestamps) > 0) {
    
    for(iindex in 1:nrow(index.timestamps)) {
        sub.index <- sprintf("%smas_index_%s.txt", data.dir, index.timestamps[iindex,])
        sub.index.size <- file.info(sub.index)$size
        if(is.na(sub.index.size)) {
            next
        }
        if(sub.index.size == 0) {
            next
        }
        
        
        pdf(sprintf("%sGraphSegments.%s.pdf", data.dir, index.timestamps[iindex,]), width=21, height=15)
        
        
        nodes.initial <- NULL
        nodes.x.initial <- NULL
        nodes.y.initial <- NULL
        edges.from.initial <- NULL
        edges.to.initial <- NULL
        

        sub.index.files <- read.csv(sub.index, header=FALSE)
        
        sub.file.is.loadfragment <- vector(mode="logical", length=nrow(sub.index.files))
        sub.file.is.loadfragment[grep(".*_LoadFragment_\\d{4}.\\d{2}.\\d{2}.\\d{2}.\\d{2}.\\d{2}.\\d{6}", sub.index.files[,1])] <- TRUE

        for(z in 1:nrow(sub.index.files)) {
            sub.index.file.name <- sub.index.files[z,1]
            sub.index.file <- paste0(sub.index.file.name, ".csv");
            sub.index.file.size <- file.info(sub.index.file)$size
            if(is.na(sub.index.file.size)) {
                next
            }
            if(sub.index.file.size == 0) {
                next
            }
            
        
            max_col <- max(count.fields(sub.index.file, sep = ','))
            
            col_names = paste0("V",seq_len(max_col))
            polygons.dataFrame <- read.csv(sub.index.file, col.names=col_names, strip.white=T, header=F, row.names = NULL, na.strings = "")
            
            
            
            N <- nrow(polygons.dataFrame) * (max_col / 3)

            nodes <- character(N)
            
            nodes.x <- double(N)
            nodes.y <- double(N)
            
            edges.from <- character(N)
            edges.to <- character(N)
            
            
            
            x <- polygons.dataFrame[, seq(1, max_col, by=3)]
            y <- polygons.dataFrame[, seq(2, max_col, by=3)]
            L <- polygons.dataFrame[, seq(3, max_col, by=3)]
            
            p <- 0
            e <- 0
            
            for(j in 1:nrow(polygons.dataFrame)) { 
                do_close_polygon <- max_col > 6
                
                a <- as.numeric(na.omit(as.numeric(x[j,])))
                b <- as.numeric(na.omit(as.numeric(y[j,])))
                
                l <- as.character(na.omit(as.character(L[j,])))
                
                
                if(do_close_polygon) {
                    a <- as.numeric(cbind(t(a), c(x[j, 1]), deparse.level = 0))
                    b <- as.numeric(cbind(t(b), c(y[j, 1]), deparse.level = 0))
                    l <- as.character(cbind(t(l), c(L[j, 1]), deparse.level = 0))
                }
                
                n <- length(a)
                
                for(k in 1:n) {
                    p <- p + 1
                    
                    nodes.x[p] <- a[k]
                    nodes.y[p] <- b[k]
                    nodes[p] <- l[k]
                    
                    if(k > 1) {
                        e <- e + 1
                        
                        edges.from[e] <- l[k - 1]
                        edges.to[e] <- l[k]
                    }
                }
            }
            
            nodes <- nodes[1:p]
            nodes.x <- nodes.x[1:p]
            nodes.y <- nodes.y[1:p]
            edges.from <- edges.from[1:e]
            edges.to <- edges.to[1:e]
            

            edge.colors <- rep("green", times=length(edges.from))
            node.colors <- rep("green", times=length(nodes))
            frame.colors <- rep("green", times=length(nodes))
            node.size <- 0.5
            node.shape <- "circle"
            node.names <- nodes
            
            
            sub.index.txt.file <- paste0(sub.index.file.name, ".txt");
            sub.index.txt.file.size <- file.info(sub.index.txt.file)$size
            
            set.initial.graph <- z==1
            
            if(!is.na(sub.index.txt.file.size) && sub.index.txt.file.size > 0) {
                plot.lines <- read.csv(sub.index.txt.file, header=FALSE)
                
                plot.lines.is.evaluate <- vector(mode="logical", length=nrow(plot.lines))
                plot.lines.is.evaluate[grep("EvaluateContours:line\\s+\\d{4}\\:\\-\\-\\-size_increment=\\-?\\d+ pass_number=\\d+ max_passes=\\d+", plot.lines[,1])] <- TRUE
                
                for(j in 1:nrow(plot.lines)) {
                    if(plot.lines.is.evaluate[j]) {
                        if(sub.file.is.loadfragment[z]) {
                            set.initial.graph = TRUE
                        }
                    }
                }
                
            }

            if(set.initial.graph) {
                nodes.initial <- paste0(nodes, "x")
                nodes.x.initial <- nodes.x
                nodes.y.initial <- nodes.y
                edges.from.initial <- paste0(edges.from, "x")
                edges.to.initial <- paste0(edges.to, "x")
                
                if(is.null(plot.lines)) {
                    plot.lines <- data.frame(V1="SET INITIAL FRAGMENT")
                }
                else {
                    plot.lines <- rbind(c("SET INITIAL FRAGMENT"),plot.lines)
                }
            }
            else {
                nodes.x <- c(nodes.x, nodes.x.initial)
                nodes.y <- c(nodes.y, nodes.y.initial)
                edges.from <- c(edges.from, edges.from.initial)
                edges.to <- c(edges.to, edges.to.initial)
                edge.colors <- c(edge.colors, rep("antiquewhite", times=length(edges.from.initial)))
                node.colors <- c(node.colors, rep("antiquewhite", times=length(nodes.initial)))
                frame.colors <- c(frame.colors, rep("antiquewhite", times=length(nodes.initial)))
                
                node.names <- c(nodes, nodes.initial) #rep(".", , times=length(nodes.initial)))
                nodes <- c(nodes, nodes.initial)
            }
            
            layout <- matrix(NA, nrow = length(nodes.x), ncol = 2)
            layout[,1] <- nodes.x
            layout[,2] <- nodes.y
            
            polygon.graph <- tbl_graph(
                nodes=data.frame(id=nodes, name=node.names, color=node.colors, frame.color=frame.colors, size=node.size, shape=node.shape),
                edges=data.frame(to=edges.to, from=edges.from, color=edge.colors), directed=TRUE)
            
            plot(polygon.graph, vertex.label.cex=0.1, edge.arrow.size=0.2, edge.arrow.width=0.5, edge.lty=1, edge.curved=FALSE, layout=layout)

            mtext(sub.index.file, side = 3, line = -1, col="darkgreen", cex=2.0, adj=1)
            
            if(is.na(sub.index.txt.file.size)) {
                next
            }
            if(sub.index.txt.file.size == 0) {
                next
            }
            
            line_offset <- 1
            for(j in 1:nrow(plot.lines)) {
                mtext(plot.lines[j,1], side = 3, line = -(j+line_offset), col="lightgrey", cex=0.9, adj=1)
                line_offset <- j + 1
            }
            
            plot.lines <- NULL
            
        }
        
        
        dev.off()
    }
}    




