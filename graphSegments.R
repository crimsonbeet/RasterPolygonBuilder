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
        
        
        polygon.graph.initial <- NULL
        layout.initial <- NULL
        
        
        sub.index.files <- read.csv(sub.index, header=FALSE)
        for(z in 1:nrow(sub.index.files)) {
            sub.index.file <- sub.index.files[z,1]
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
                
                # n <- length(x[j,])
                # 
                # if(is.na(x[j, n])) {
                #     x[j, n] = x[j, 1]
                #     y[j, n] = y[j, 1]
                #     
                #     do_close_polygon <- FALSE
                # } 

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
            

            edge.colors <- "green"
            node.colors <- "green"
            node.size <- 0.5
            node.shape <- "circle"
            
            polygon.graph <- tbl_graph(
                nodes=data.frame(id=nodes, name=nodes, color=node.colors, frame.color=node.colors, size=node.size, shape=node.shape),
                edges=data.frame(to=as.character(edges.to), from=as.character(edges.from), color=edge.colors), directed=TRUE)
            
            if(z == 1) {
                max.x <- max(nodes.x)
                min.x <- min(nodes.x)
                max.y <- max(nodes.y)
                min.y <- min(nodes.y)
                
                offset.x <- (max.x - min.x) / 2
                offset.y <- (max.y - min.y) / 2
                
                scale.x <- 7.5 / (max.x - offset.x)
                scale.y <- 7.5 / (max.y - offset.y)
                
                box.x <- c(min.x, min.x, max.x, max.x)
                box.x <- box.x- offset.x
                box.x <- box.x * scale.x
                box.y <- c(min.y, min.y, max.y, max.y)
                box.y <- box.y- offset.y
                box.y <- box.y * scale.y
            }
            
            offset.x <- 0
            offset.y <- 0
            
            scale.x <- 1
            scale.y <- 1

            layout <- matrix(NA, nrow = p, ncol = 2)
            layout[,1] <- (nodes.x - offset.x) * scale.x
            layout[,2] <- (nodes.y - offset.y) * scale.y
            
            add_mode <- FALSE
            if(is.null(polygon.graph.initial)) {
                polygon.graph.initial <- tbl_graph(
                    nodes=data.frame(id=nodes, name=nodes, color="antiquewhite", frame.color="antiquewhite", size=node.size, shape=node.shape),
                    edges=data.frame(to=as.character(edges.to), from=as.character(edges.from), color="antiquewhite"), directed=TRUE)
                layout.initial <- layout
            }
            else {
                plot(polygon.graph.initial, vertex.label=NA, edge.arrow.size=0.1, edge.arrow.width=0.2, edge.lty=1, edge.curved=FALSE, layout=layout.initial)
                add_mode <- TRUE
            }
            
            plot(polygon.graph, vertex.label.cex=0.1, edge.arrow.size=0.2, edge.arrow.width=0.5, edge.lty=1, edge.curved=FALSE, layout=layout, add=add_mode)

            mtext(sub.index.file, side = 3, line = -1, col="darkgreen", cex=2.0, adj=1)
        }
        
        
        dev.off()
    }
}    









file_name <- "2022Aug26_185105_397_ClassifyBits0.csv"
file_name <- "2024Apr26_231459_417_ClassifyBits0.csv"




max_col <- max(count.fields(file_name, sep = ','))
col_names = paste0("V",seq_len(max_col))
size_segments <- read.csv(file_name, col.names=col_names, strip.white=T, header=F, row.names = NULL)


N <- nrow(size_segments) * 2
N2 <- N/2


nodes <- character(N)

nodes.x <- double(N)
nodes.y <- double(N)

edges.from <- character(N2)
edges.to <- character(N2)


x <- size_segments[, seq(1, max_col, by=2)]
y <- size_segments[, seq(2, max_col, by=2)]


for(j in 1:N2) { 
    if(is.na(x[j,length(x[j,])])) {
        x[j,length(x[j,])] = x[j, 1]
        y[j,length(x[j,])] = y[j, 1]
    } 

    a <- as.numeric(na.omit(as.numeric(x[j,])))
    b <- as.numeric(na.omit(as.numeric(y[j,])))
    
    p <- c(2*j-1, 2*j)
    
    for(z in 1:2) {
        nodes.x[p[z]] <-a[z]
        nodes.y[p[z]] <-b[z]
        nodes[p[z]] <- ifelse(z==2, sprintf("%d", p[z]), as.character(p[z]))
    }
    
    edges.from[j] <- nodes[p[1]]
    edges.to[j] <- nodes[p[2]]
}


edge.colors <- "green"
node.colors <- "green"
node.size <- 1
node.shape <- "circle"

gas.panel2 <- tbl_graph(
    nodes=data.frame(id=nodes, color=node.colors, size=node.size, shape=node.shape),
    edges=data.frame(to=as.character(edges.to), from=as.character(edges.from), color=edge.colors), directed=TRUE)

max.x <- max(nodes.x)
min.x <- min(nodes.x)
max.y <- max(nodes.y)
min.y <- min(nodes.y)

offset.x <- (max.x - min.x) / 2
offset.y <- (max.y - min.y) / 2

scale.x <- 7.5 / (max.x - offset.x)
scale.y <- 7.5 / (max.y - offset.y)

layout <- matrix(NA, nrow = N, ncol = 2)
layout[,1] <- (nodes.x - offset.x) * scale.x
layout[,2] <- (nodes.y - offset.y) * scale.y




pdf("GrapSegments.pdf", width=21, height=15)
plot(gas.panel2, vertex.label.cex=0.3, edge.arrow.size=0.2, edge.arrow.width=0.5, edge.lty=1, edge.curved=FALSE, layout=layout)
dev.off()







nodes <- read.csv(sprintf("%s%s", data.dir, "GraphModuleNodes.csv"), header=TRUE)
if(length(which(is.na(nodes$id))) > 0)
    nodes <- nodes[-which(is.na(nodes$id)),]
nodes <- data.frame(id=nodes$id, name=nodes$name, dev.type=nodes$dev, dev.port=nodes$is.port)

edges <- read.csv(sprintf("%s%s", data.dir, "GraphModuleEdges.csv"), header=TRUE)



edge.colors <- "green"




TRANSDDUCER <- 6
VALVE <- 3
GASSOURCE <- 1
PURGESOURCE <- 2
PIPE <- 10
MFC <- 4
Lfc <- 5



node.colors <- ifelse(nodes$dev.type==PIPE, "blue",
                      ifelse(nodes$dev.type==TRANSDUCER, "antiquewhite",
                             ifelse(nodes$dev.port==TRUE, "magenta",
                                    ifelse(nodes$dev.type==PURGESOURCE, "brown", "red"))))
node.size <- ifelse(nodes$dev.type==PIPE, 5, ifelse(nodes$dev.port==TRUE, 7, 2))
node.shape <- ifelse(nodes$dev.type==VALVE, "square", ifelse(nodes$dev.type==GASSOURCE, "crectangle", ifelse(nodes$dev.type==PURGESOURCE, "crectangle", "circle")))


gas.panel2 <- tbl_graph(
    nodes=data.frame(id=as.character(nodes$id), name=nodes$name, dev.type=nodes$dev.type, color=node.colors, size=node.size, shape=node.shape),
    edges=data.frame(to=as.character(edges$to), from=as.character(edges$from), color=edge.colors), directed=TRUE)


pdf("GraphModuleRnode.pdf", width=21, height=15)
plot(gas.panel2, vertex.label.cex=0.8, edge.arrow.size=1, edge.arrow.width=1, edge.lty=1, edge.curved=FLASE)
dev.off()


gas.panel <- tbl_graph(nodes=data.frame(name=nodes$name, dev.type=nodes$dev.type), edges=edges, directed=TRUE)
node.colors <- ifelse(nodes$dev.type==PIPE, "blue",
                      ifelse(nodes$dev.type==TRANSDUCER, "antiquewhite",
                             ifelse(nodes$dev.port==TRUE, "magenta",
                                    ifelse(nodes$dev.type==PURGESOURCE, "brown", "red"))))
node.size <- ifelse(nodes$dev.type==PIPE, 5, ifelse(nodes$dev.port==TRUE, 7, 2))


ggraph(gas.panel, layout="stress") + 
    geom_node_point(size=node.size, colour=node.colors) +
    geom_node_text(aes(label=name)) +
    geom_edge_link(arrow=arrow(length=unit(0.1, "inches"), type="closed"), colour="green") +
    theme_graph()


create.graph <- function() {
    if(length(which(is.na(nodes$id))) > 0)
        nodes <- nodes[-which(is.na(nodes$id)),]
    if(length(which((nodes$id==-1))) > 0)
        nodes <- nodes[-which(nodes$id==-1),]
    
    edge.colors >- "antiquewhite"
    node.colors >- "antiquewhite"
    node.size <- ifelse(nodes$dev.type==PIPE, 5, ifelse(nodes$dev.type==GASSOURCE, 1.5, ifelse(nodes$dev.port==TRUE, 7, 2)))
    node.shape <- ifelse(nodes$dev.type==VALVE, "square", ifelse(nodes$dev.type==GASSOURCE, "crectangle", ifelse(nodes$dev.type==PURGESOURCE, "crectangle", "circle")))

    gas.panel2 <- tbl_graph(
        nodes=data.frame(id=as.character(nodes$id), name=nodes$name, dev.type=nodes$dev.type, color=node.colors, size=node.size, shape=node.shape), node_key="id",
        edges=data.frame(to=as.character(edges$to), from=as.character(edges$from), color=edge.colors), directed=TRUE)
    
    obj <- list()
    
    obj$gas.panel <- gas.panel
    
    obj
}



bit.masks <- c(1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768)
bit.colors <- c("green", "red", "blue", "yellow", "cyan", "brown", "maroon3","palegreen","limegreen","seagreen","green", "red", "blue", "yellow", "cyan", "brown", "maroon3")
getColor <- function(bit.mask) {
    color <- NA
    if(bit.mask==0) {
        color <- "antiquewhite"
    }
    else
    if(bit.mask %in% bit.masks) {
        color <- bit.colors[log(bit.mask, 2) + 1]
    }
    else {
        color <- "orange"
    }
    color
}


state.masks <- c(1,2,4,8,16)
state.texts <- c("absent", "pressurizing", "flowinf", "evacuating", "isolated")
state.colors <- c("red", "brown", "blue", "seagreen", "cyan")
getStateColor <- function(state.mask) {
    color <- NA
    if(state.mask %in% state.masks) {
        color <- state.colors[log(state.mask, 2) + 1]
    }
    else {
        color <- "orange"
    }
    color
}

plotStateLegend <- function() {
    for(j in 1:length(state.texts)) {
        mtext(state.texts[j], side=3, line=-j, col=state.colors[j], adj=1)
    }
}




gas.panel <- NULL

bitmask.nodes <- NULL
bitmask.edges <- NULL

enabled.nodes <- NULL



graph.chamber <- create.graph()


N <- max(nodes$id)
if(is.na(N)) {
    N <- nrow(nodes)
}


gas.panel <- graph.chamber$gas.panel

bitmask.nodes <- integer(N)
bitmask.edges <- integer(nrow(edges))


enabled.nodes <- logical(N)
enabled.nodes[nodes[nodes$dev.type==PIPE,]$id] <- TRUE




test.traces <- as.character()


index <- read.csv(sprintf("%s%s", data.dir, "gmu.index.txt"), header=FALSE)

if(nrow(index) > 0) {
    
    
    index.is.static <- vector(mode="logical", length=nrow(index))
    index.is.static[grep("\\d\\d-\\d\\d-\\d\\d\\d\\d\\.\\d\\d-\\d\\d-\\d\\d\\.\\d{6}.static", index[,1])] <- TRUE
    
    substring.datetime <- strsplit(index[1,1], "\\.\\d{6}.")[[1]][1]
    
    nodesStates.dataFrame <- NULL
    
    pdf(sprintf("GraphModule.%s.TestRun.pdf", substring.datetime), width=21, height=15)
    
    for(z in 1:nrow(index)) {
        file.name.pref <- sprintf("%sgmu.%s", data.dir, index[z,1])
        
        edgesBitmasks.dataFrame <- read.csv(sprintf("%s.EdgesBitmasks.csv", file.name.pref), header=TRUE)
        nodesBitmasks.dataFrame <- read.csv(sprintf("%s.NodesBitmasks.csv", file.name.pref), header=TRUE)
        
        statesSize <- file.info(sprintf("%s.NodesStates.csv", file.name.pref))$size
        
        
        if(!is.na(statesSize)) {
            if(statesSize > 0) {
                nodesStates.dataFrame <- read.csv(sprintf("%s.NodesStates.csv", file.name.pref), header=TRUE)
            }
        }
        else {
            statesSize <- 0
        }
        
        
        bitmask.edges <- edgesBitmasks.dataFrame$bitmask
        bitmask.nodes <- nodesBitmasks.dataFrame$bitmask
        
        
        for(x in 1:length(bitmask.nodes)) {
            V(gas.panel)[ as.integer(rownames(nodes)[nodes$id==x]) ]$color <- getColor(bitmask.nodes[x])
        }
        for(x in 1:length(bitmask.edges)) {
            E(gas.panel)[x]$color <- getColor(bitmask.edges[x])
        }
        
        
        set.seed(1)
        layout <- layout_nicely(gas.panel)
        plot(gas.panel, vertex.label=NA, edge.arrow.size=0, edge.arrow.width=0, edge.lty=1, layout=layout, vertex.shape="none")
        
        edges.directional <- which(edgesBitmasks.dataFrame$directional)
        graph.directional <- delete_edges(gas.panel, E(gas.panel)[(1:ecount(gas.panel))[-edges.directional]])
        
        plot(graph.directional, vertex.label=NA, edge.arrow.size=1, edge.arrow.width=1, edge.lty=1, layout=layout, vertex.shape="none", add=TRUE)
        plot(gas.panel, vertex.label.cex=0.5, edge.arrow.size=0, edge.lty=0, layout=layout, add=TRUE)
        
        
        
        if(index.is.static[z]) {
            line_offset <- 0
            
            if(file.info(sprintf("%s.Plot.txt", file.name.pref))$size > 0) {
                plot.lines <- read.csv(sprintf("%s.Plot.txt", file.name.pref), header=FALSE)
                
                for(j in 1:nrow(plot.lines)) {
                    mtext(plot.lines[j,1], side = 3, line = -j, col="darkgreen", cex=0.0, adj=1)
                    line_offset <- j
                    test.traces[length(test.traces) + 1] <- plot.lines[j,1]
                }
            }
            
            if(file.info(sprintf("%s.Trace.txt", file.name.pref))$size > 0) {
                plot.lines <- read.csv(sprintf("%s.Trace.txt", file.name.pref), header=FALSE)
                
                for(j in 1:nrow(plot.lines)) {
                    mtext(plot.lines[j,1], side = 3, line = -(j+line_offset), col="lightgrey", cex=0.9, adj=1)
                    line_offset <- j
                    test.traces[length(test.traces) + 1] <- plot.lines[j,1]
                }
            }
        }
        else {
            plot.lines <- read.csv(sprintf("%s.Plot.txt", file.name.pref), header=FALSE)
            
            for(j in 1:nrow(plot.lines)) {
                mtext(plot.lines[j,1], side = 3, line = -j, col="darkgreen", cex=0.0, adj=1)
                line_offset <- j
                test.traces[length(test.traces) + 1] <- plot.lines[j,1]
            }
        }
        
        
        
        if(statesSize > 0) {
            x <- 1
            for(j in 1:nrow(nodesStates.dataFrame)) {
                z <- nodesStates.dataFrame$node[j]
                if(x < z) {
                    for(y in x:(z-1)) {
                        V(gas.panel)[y]$color <- "antiquewhite"
                    }
                }
                V(gas.panel[z])$color <- getStateColor(nodesStates.dataFrame$state[j])
                x <- z+1
            }
            set.seed(1)
            plot(gas.panel, vertex.label.cex=0.5, edge.arrow.size=1, edge.arrow.width=1, edge.lty=1, layout=layout)
            plotStateLegend()
            
        }
        
    }
    
    dev.off()
    
    file.remove(sprintf("%s%s", data.dir, "gmu.index.txt"))
    
    
    test.traces[length(test.traces) + 1] <- "END"
    test.traces[length(test.traces) + 1] <- paste("nodesBitmasks:", as.character(nodesBitmasks.dataFrame)[1], sep="")
    test.traces[length(test.traces) + 1] <- paste("edgesBitmasks:", as.character(edgesBitmasks.dataFrame)[1], sep="")
    

    write.table(test.traces, sprintf("%s%s", file.name.pref, ".Usecase.txt"), row,names=FALSE, col.names=FALSE)
}








