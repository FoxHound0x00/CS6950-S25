<template>
  <div class="force-directed-graph">
    <h3>Cluster Visualization</h3>
    <div ref="graphContainer" class="graph-container"></div>
    <!-- <div class="legend" ref="legendContainer"></div> -->
  </div>
</template>

<script>
import { defineComponent, ref, onMounted, watch, onBeforeUnmount } from 'vue';
import * as d3 from 'd3';

export default defineComponent({
  name: 'ForceDirectedGraph',
  props: {
    data: {
      type: Object,
      required: true
    }
  },
  setup(props) {
    const graphContainer = ref(null);
    const legendContainer = ref(null);
    const svg = ref(null);
    const width = ref(800);
    const height = ref(600);
    
    // Extend color scheme to support more clusters
    const extendedColorScheme = [
      ...d3.schemeCategory10,
      "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", 
      "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd"
    ];
    
    // Create the visualization
    const createVisualization = () => {
      if (!graphContainer.value) return;
      
      // Clear any previous visualization
      d3.select(graphContainer.value).selectAll("*").remove();
      
      // Get container dimensions
      width.value = graphContainer.value.clientWidth || 800;
      height.value = graphContainer.value.clientHeight || 600;
      
      // Create SVG
      svg.value = d3.select(graphContainer.value)
        .append('svg')
        .attr('width', width.value)
        .attr('height', height.value)
        .attr('viewBox', [0, 0, width.value, height.value])
        .attr('style', 'max-width: 100%; height: auto;');
      
      // Create clusters and outlier blobs with more diversity
      const { clusters, outliersBlobs } = generateDiverseClustersWithOutliers();
      
      // Group for hulls (behind nodes)
      const hullGroup = svg.value.append('g').attr('class', 'hulls');
      
      // Group for links
      const linkGroup = svg.value.append('g').attr('class', 'links');
      
      // Group for nodes (on top)
      const nodeGroup = svg.value.append('g').attr('class', 'nodes');
      
      // Draw links between nodes in main clusters
      const links = [];
      clusters.forEach(cluster => {
        for (let i = 0; i < cluster.nodes.length; i++) {
          // Connect to a subset of nodes in the same cluster
          for (let j = i + 1; j < cluster.nodes.length; j++) {
            if (Math.random() > 0.7) { // Only create some links, not all possible ones
              links.push({
                source: cluster.nodes[i],
                target: cluster.nodes[j],
                cluster: cluster.id,
                isOutlier: false
              });
            }
          }
        }
      });
      
      // Draw links within outlier blobs
      outliersBlobs.forEach(blob => {
        for (let i = 0; i < blob.nodes.length; i++) {
          // Create denser connections within outlier blobs
          for (let j = i + 1; j < blob.nodes.length; j++) {
            if (Math.random() > 0.5) { // More links in outlier blobs
              links.push({
                source: blob.nodes[i],
                target: blob.nodes[j],
                cluster: blob.parentClusterId, // Original cluster color for links
                isOutlier: true
              });
            }
          }
        }
      });
      
      // Draw occasional links between outlier blobs that are close to each other
      for (let i = 0; i < outliersBlobs.length; i++) {
        for (let j = i + 1; j < outliersBlobs.length; j++) {
          // Check distance between blob centers
          const dx = outliersBlobs[i].center.x - outliersBlobs[j].center.x;
          const dy = outliersBlobs[i].center.y - outliersBlobs[j].center.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          // Only connect if relatively close and with some probability
          if (distance < outliersBlobs[i].radius + outliersBlobs[j].radius + 40 && Math.random() > 0.7) {
            // Connect a few nodes between these blobs
            const nodePairsToConnect = Math.min(2, Math.floor(Math.random() * 3) + 1);
            
            for (let k = 0; k < nodePairsToConnect; k++) {
              const sourceIndex = Math.floor(Math.random() * outliersBlobs[i].nodes.length);
              const targetIndex = Math.floor(Math.random() * outliersBlobs[j].nodes.length);
              
              links.push({
                source: outliersBlobs[i].nodes[sourceIndex],
                target: outliersBlobs[j].nodes[targetIndex],
                cluster: outliersBlobs[i].parentClusterId,
                isOutlier: true,
                isBridge: true
              });
            }
          }
        }
      }
      
      // Draw links
      linkGroup.selectAll('line')
        .data(links)
        .join('line')
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y)
        .attr('stroke', d => extendedColorScheme[d.cluster % extendedColorScheme.length])
        .attr('stroke-opacity', d => d.isOutlier ? 0.3 : 0.5)
        .attr('stroke-width', d => d.isBridge ? 0.5 : 1)
        .attr('stroke-dasharray', d => d.isBridge ? '3,3' : 'none');
      
      // Draw nodes
      const allNodes = [
        ...clusters.flatMap(cluster => cluster.nodes),
        ...outliersBlobs.flatMap(blob => blob.nodes)
      ];
      
      const nodeCirlces = nodeGroup.selectAll('circle')
        .data(allNodes)
        .join('circle')
        .attr('cx', d => d.x)
        .attr('cy', d => d.y)
        .attr('r', d => d.isOutlier ? 3 : 5)
        .attr('fill', d => extendedColorScheme[d.originalCluster % extendedColorScheme.length])
        .attr('stroke', '#fff')
        .attr('stroke-width', d => d.isOutlier ? 1 : 1.5)
        .attr('fill-opacity', d => d.isOutlier ? 0.8 : 1)
        .attr('cursor', 'grab');
      
      // Draw hulls around main clusters
      const clusterHulls = clusters.map(cluster => {
        // Create points array for hull
        const points = cluster.nodes.map(node => [node.x, node.y]);
        
        // Calculate convex hull
        const hull = d3.polygonHull(points);
        
        // Draw hull path with padding
        if (hull) {
          // Add some padding around the hull
          const paddedHull = padHull(hull, 15);
          
          return {
            clusterId: cluster.id,
            element: hullGroup.append('path')
              .attr('d', `M${paddedHull.join('L')}Z`)
              .attr('fill', extendedColorScheme[cluster.id % extendedColorScheme.length])
              .attr('fill-opacity', 0.2)
              .attr('stroke', extendedColorScheme[cluster.id % extendedColorScheme.length])
              .attr('stroke-width', 2)
              .attr('stroke-opacity', 0.5)
          };
        }
        return null;
      }).filter(Boolean);
      
      // Draw lighter hulls around outlier blobs
      const blobHulls = outliersBlobs.map(blob => {
        // Create points array for hull
        const points = blob.nodes.map(node => [node.x, node.y]);
        
        // Calculate convex hull
        const hull = d3.polygonHull(points);
        
        // Draw hull path with padding
        if (hull && hull.length > 2) { // Need at least 3 points for a hull
          // Add some padding around the hull
          const paddedHull = padHull(hull, 8);
          
          return {
            blobId: blob.id,
            element: hullGroup.append('path')
              .attr('d', `M${paddedHull.join('L')}Z`)
              .attr('fill', extendedColorScheme[blob.displayClusterId % extendedColorScheme.length])
              .attr('fill-opacity', 0.1)
              .attr('stroke', extendedColorScheme[blob.displayClusterId % extendedColorScheme.length])
              .attr('stroke-width', 1)
              .attr('stroke-opacity', 0.3)
              .attr('stroke-dasharray', '3,3')
          };
        }
        return null;
      }).filter(Boolean);
      
      // Create legend
      createLegendWithDiverseClusters(clusters, outliersBlobs);
      
      // Create force simulation for drag interactions
      const simulation = d3.forceSimulation(allNodes)
        .force('x', d3.forceX(d => d.initialX).strength(0.02))
        .force('y', d3.forceY(d => d.initialY).strength(0.02))
        .force('collide', d3.forceCollide().radius(d => d.isOutlier ? 4 : 8))
        .force('link', d3.forceLink(links).id(d => d.id).strength(0.1))
        .on('tick', () => {
          // Update node positions
          nodeCirlces
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
            
          // Update link positions
          linkGroup.selectAll('line')
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
            
          // Update hull positions
          updateHulls(clusters, clusterHulls, outliersBlobs, blobHulls);
        })
        .alphaDecay(0.01);
      
      // Add drag behavior
      nodeCirlces.call(d3.drag()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
          d3.select(event.sourceEvent.target).attr('cursor', 'grabbing');
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
          d3.select(event.sourceEvent.target).attr('cursor', 'grab');
        }));
      
      // Enable zoom and pan
      svg.value.call(d3.zoom()
        .extent([[0, 0], [width.value, height.value]])
        .scaleExtent([0.5, 5])
        .on('zoom', zoomed));
        
      function zoomed(event) {
        nodeGroup.attr('transform', event.transform);
        linkGroup.attr('transform', event.transform);
        hullGroup.attr('transform', event.transform);
      }
    };
    
    // Update hulls when nodes move
    function updateHulls(clusters, clusterHulls, outliersBlobs, blobHulls) {
      // Update main cluster hulls
      clusters.forEach((cluster, i) => {
        if (clusterHulls[i]) {
          const points = cluster.nodes.map(node => [node.x, node.y]);
          const hull = d3.polygonHull(points);
          
          if (hull) {
            const paddedHull = padHull(hull, 15);
            clusterHulls[i].element.attr('d', `M${paddedHull.join('L')}Z`);
          }
        }
      });
      
      // Update outlier blob hulls
      outliersBlobs.forEach((blob, i) => {
        if (blobHulls[i]) {
          const points = blob.nodes.map(node => [node.x, node.y]);
          const hull = d3.polygonHull(points);
          
          if (hull && hull.length > 2) {
            const paddedHull = padHull(hull, 8);
            blobHulls[i].element.attr('d', `M${paddedHull.join('L')}Z`);
          }
        }
      });
    }
    
    // Function to pad hull points outward
    function padHull(hull, padding) {
      if (!hull || hull.length === 0) return [];
      
      // Calculate centroid
      const centroid = hull.reduce(
        (acc, point) => [acc[0] + point[0], acc[1] + point[1]],
        [0, 0]
      );
      centroid[0] /= hull.length;
      centroid[1] /= hull.length;
      
      // Pad points outward from centroid
      return hull.map(point => {
        const dx = point[0] - centroid[0];
        const dy = point[1] - centroid[1];
        const len = Math.sqrt(dx * dx + dy * dy);
        const scale = (len + padding) / len;
        
        return [
          centroid[0] + dx * scale,
          centroid[1] + dy * scale
        ];
      });
    }
    
    // Generate diverse clusters with many outlier blobs
    const generateDiverseClustersWithOutliers = () => {
      const clusters = [];
      const outliersBlobs = [];
      
      // Create main cluster positions in a grid-like pattern with some randomness
      const mainClusterCount = 5; // Original 5 clusters as base
      const positions = [];
      
      // Create position grid for main clusters
      const gridSize = Math.ceil(Math.sqrt(mainClusterCount));
      const cellWidth = width.value / (gridSize + 1);
      const cellHeight = height.value / (gridSize + 1);
      
      for (let i = 0; i < mainClusterCount; i++) {
        const row = Math.floor(i / gridSize);
        const col = i % gridSize;
        
        positions.push({
          x: cellWidth * (col + 1) + (Math.random() * 60 - 30),
          y: cellHeight * (row + 1) + (Math.random() * 60 - 30)
        });
      }
      
      // Create main clusters
      for (let i = 0; i < mainClusterCount; i++) {
        const center = positions[i];
        const radius = Math.random() * 30 + 40; // Random size between 40-70
        const nodeCount = Math.floor(Math.random() * 15) + 15; // 15-30 nodes
        
        const nodes = [];
        
        // Generate nodes for this cluster
        for (let j = 0; j < nodeCount; j++) {
          // Random angle
          const angle = Math.random() * Math.PI * 2;
          // Random distance from center (with some variation)
          const distance = Math.random() * radius;
          
          // Calculate position
          const x = center.x + Math.cos(angle) * distance;
          const y = center.y + Math.sin(angle) * distance;
          
          nodes.push({
            id: `node-${i}-${j}`,
            x,
            y,
            initialX: x,
            initialY: y,
            originalCluster: i,
            currentCluster: i,
            isOutlier: false
          });
        }
        
        clusters.push({
          id: i,
          nodes,
          center,
          radius
        });
      }
      
      // Now create diverse outlier blobs (up to 20 total)
      // Some will share original cluster colors, some will have new colors
      const totalOutlierBlobCount = Math.floor(Math.random() * 8) + 12; // 12-20 blobs
      const maxColorId = 19; // Up to 20 different colors
      
      for (let i = 0; i < totalOutlierBlobCount; i++) {
        // Randomly determine characteristics of this blob
        const isNewColorBlob = Math.random() > 0.5; 
        const belongsToOriginalCluster = Math.random() > 0.5;
        
        // Determine blob's original/parent cluster
        const parentClusterId = Math.floor(Math.random() * mainClusterCount);
        
        // Determine display cluster (can be different)
        let displayClusterId;
        if (isNewColorBlob) {
          // Assign a new color beyond the original clusters
          displayClusterId = mainClusterCount + Math.floor(Math.random() * (maxColorId - mainClusterCount));
        } else if (belongsToOriginalCluster) {
          // Same color as parent
          displayClusterId = parentClusterId;
        } else {
          // Different color from parent but from original clusters
          displayClusterId = (parentClusterId + 1 + Math.floor(Math.random() * (mainClusterCount - 1))) % mainClusterCount;
        }
        
        // Position the blob - some close to their parent cluster, some in between clusters, some isolated
        let blobCenter;
        const positioning = Math.random();
        
        if (positioning < 0.4) {
          // Close to parent cluster
          const parentCenter = clusters[parentClusterId].center;
          const parentRadius = clusters[parentClusterId].radius;
          const angle = Math.random() * Math.PI * 2;
          const distance = parentRadius + (Math.random() * parentRadius * 0.8);
          
          blobCenter = {
            x: parentCenter.x + Math.cos(angle) * distance,
            y: parentCenter.y + Math.sin(angle) * distance
          };
        } else if (positioning < 0.7) {
          // Between two clusters
          const cluster1 = Math.floor(Math.random() * mainClusterCount);
          let cluster2;
          do {
            cluster2 = Math.floor(Math.random() * mainClusterCount);
          } while (cluster2 === cluster1);
          
          const center1 = clusters[cluster1].center;
          const center2 = clusters[cluster2].center;
          const ratio = 0.3 + Math.random() * 0.4; // Between the two clusters
          
          blobCenter = {
            x: center1.x + (center2.x - center1.x) * ratio,
            y: center1.y + (center2.y - center1.y) * ratio,
          };
        } else {
          // Isolated position
          blobCenter = {
            x: Math.random() * (width.value * 0.8) + width.value * 0.1,
            y: Math.random() * (height.value * 0.8) + height.value * 0.1
          };
        }
        
        // Blob size (smaller than main clusters)
        const blobRadius = Math.random() * 25 + 15; // 15-40 radius
        
        // Number of nodes in this blob
        const blobNodeCount = Math.floor(Math.random() * 5) + 4; // 4-9 nodes
        
        const blobNodes = [];
        
        // Generate nodes for this outlier blob
        for (let j = 0; j < blobNodeCount; j++) {
          // Random angle
          const nodeAngle = Math.random() * Math.PI * 2;
          // Random distance from blob center
          const nodeDistance = Math.random() * blobRadius;
          
          // Calculate position
          const x = blobCenter.x + Math.cos(nodeAngle) * nodeDistance;
          const y = blobCenter.y + Math.sin(nodeAngle) * nodeDistance;
          
          // Decide node color
          let nodeOriginalCluster;
          
          if (Math.random() > 0.2) {
            // Most nodes match the blob's display cluster
            nodeOriginalCluster = displayClusterId;
          } else {
            // Some nodes have random colors (mixed membership)
            nodeOriginalCluster = Math.floor(Math.random() * maxColorId);
          }
          
          blobNodes.push({
            id: `outlier-${i}-${j}`,
            x,
            y,
            initialX: x,
            initialY: y,
            originalCluster: nodeOriginalCluster,
            currentCluster: displayClusterId,
            isOutlier: true,
            blobId: i
          });
        }
        
        outliersBlobs.push({
          id: i,
          parentClusterId,
          displayClusterId,
          nodes: blobNodes,
          center: blobCenter,
          radius: blobRadius
        });
      }
      
      return { clusters, outliersBlobs };
    };
    
    // Create legend with diverse clusters and outlier blobs
    const createLegendWithDiverseClusters = (clusters, outliersBlobs) => {
      if (!legendContainer.value) return;
      
      // Clear previous legend
      d3.select(legendContainer.value).selectAll("*").remove();
      
      // Get unique display cluster IDs
      const uniqueClusterIds = new Set();
      
      // Add main clusters
      clusters.forEach(cluster => uniqueClusterIds.add(cluster.id));
      
      // Add outlier blob display clusters
      outliersBlobs.forEach(blob => uniqueClusterIds.add(blob.displayClusterId));
      
      // Convert to array and sort
      const clusterIds = Array.from(uniqueClusterIds).sort((a, b) => a - b);
      
      const legendSvg = d3.select(legendContainer.value)
        .append('svg')
        .attr('width', '100%')
        .attr('height', clusterIds.length * 20 + 10);
        
      const legend = legendSvg.selectAll('.legend-item')
        .data(clusterIds)
        .enter()
        .append('g')
        .attr('class', 'legend-item')
        .attr('transform', (d, i) => `translate(10, ${i * 20 + 10})`);
        
      legend.append('circle')
        .attr('r', 6)
        .attr('fill', d => extendedColorScheme[d % extendedColorScheme.length]);
        
      legend.append('text')
        .attr('x', 12)
        .attr('y', 4)
        .text(d => {
          // Check if it's a main cluster
          const isMainCluster = d < clusters.length;
          if (isMainCluster) {
            return `Cluster ${d + 1}`;
          } else {
            return `Cluster ${d + 1} (Outlier Cluster)`;
          }
        });
    };
    
    // Handle window resize
    const handleResize = () => {
      if (!graphContainer.value) return;
      createVisualization();
    };
    
    // Clean up event listeners on unmount
    onBeforeUnmount(() => {
      window.removeEventListener('resize', handleResize);
    });
    
    // Initialize on mount
    onMounted(() => {
      window.addEventListener('resize', handleResize);
      createVisualization();
    });
    
    // Recreate visualization when data changes
    watch(() => props.data, () => {
      createVisualization();
    }, { deep: true });
    
    return {
      graphContainer,
      legendContainer
    };
  }
});
</script>

<style scoped>
.force-directed-graph {
  margin-bottom: 30px;
  background-color: #f8f9fa;
  padding: 20px;
  border-radius: 8px;
  border: 1px solid #e9ecef;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.force-directed-graph h3 {
  margin-top: 0;
  margin-bottom: 15px;
  font-size: 16px;
  color: #2c3e50;
  font-weight: 600;
}

.graph-container {
  width: 100%;
  height: 500px;
  background-color: white;
  border-radius: 4px;
  overflow: hidden;
  border: 1px solid #dcdfe6;
  margin-bottom: 20px;
}

.legend {
  background-color: white;
  padding: 10px;
  border-radius: 4px;
  border: 1px solid #dcdfe6;
  max-height: 150px;
  overflow-y: auto;
}

svg {
  display: block;
  max-width: 100%;
}
</style>