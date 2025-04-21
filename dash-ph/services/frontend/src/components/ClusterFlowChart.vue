<template>
    <div>
      <div ref="chart" class="chart-container"></div>
      <div class="info-box" ref="infoBox">
        <div v-if="selectedInfo">
          <h3>{{ selectedInfo.title }}</h3>
          <div class="info-stats">
            <div class="stat-item">
              <div class="stat-value">{{ selectedInfo.count }}</div>
              <div class="stat-label">Items</div>
            </div>
            <div class="stat-item">
              <div class="stat-value">{{ selectedInfo.incoming }}</div>
              <div class="stat-label">Incoming</div>
            </div>
            <div class="stat-item">
              <div class="stat-value">{{ selectedInfo.outgoing }}</div>
              <div class="stat-label">Outgoing</div>
            </div>
            <div class="stat-item">
              <div class="stat-value">{{ selectedInfo.total }}</div>
              <div class="stat-label">Total</div>
            </div>
          </div>
        </div>
        <div v-else class="empty-info">
          Hover over a cluster to see details
        </div>
      </div>
    </div>
  </template>
  <script>
  import * as d3 from 'd3';
  export default {
    name: 'ClusterFlowChart',
    props: {
      data: {
        type: Object,
        required: true,
        default: () => ({})
      },
      width: {
        type: Number,
        default: 800
      },
      height: {
        type: Number,
        default: 500
      }
    },
    data() {
      return {
        selectedInfo: null
      };
    },
    mounted() {
      this.renderChart();
    },
    watch: {
      data: {
        handler() {
          this.renderChart();
        },
        deep: true
      }
    },
    methods: {
      // Helper function to escape CSS selectors
      escapeSelector(selector) {
        return selector.replace(/[ !"#$%&'()*+,.:;<=>?@[\\\]^`{|}~]/g, '\\$&');
      },
      // Update the info box with cluster data
      updateInfoBox(stage, cluster, count, connections, isSelected = false) {
        this.selectedInfo = {
          title: `${stage} - Cluster ${cluster}${isSelected ? ' (Selected)' : ''}`,
          count: count,
          incoming: connections.incoming,
          outgoing: connections.outgoing,
          total: connections.total
        };
      },
      // Clear the info box
      clearInfoBox() {
        this.selectedInfo = null;
      },
      renderChart() {
        const el = this.$refs.chart;
        if (!el || !this.data || Object.keys(this.data).length === 0) {
          return;
        }
        
        el.innerHTML = '';
        
        const { width, height } = this;
        const margin = { top: 30, right: 50, bottom: 30, left: 50 };
        const innerWidth = width - margin.left - margin.right;
        const innerHeight = height - margin.top - margin.bottom;
        
        // Get stages (keys from data)
        const stages = Object.keys(this.data);
        if (stages.length < 2) return;
        
        // Create SVG
        const svg = d3.select(el)
          .append('svg')
          .attr('width', width)
          .attr('height', height);
        
        // Add a group for the chart
        const chart = svg.append('g')
          .attr('transform', `translate(${margin.left}, ${margin.top})`);
        
        // Find all unique clusters across all stages
        const uniqueClusters = new Set();
        stages.forEach(stage => {
          if (Array.isArray(this.data[stage])) {
            this.data[stage].forEach(cluster => {
              uniqueClusters.add(cluster);
            });
          }
        });
        
        // Convert to array and sort
        const clusters = Array.from(uniqueClusters).sort();
        
        // Create color scale
        const colorScale = d3.scaleOrdinal(d3.schemeCategory10)
          .domain(clusters);
        
        // Create x scale for stages
        const xScale = d3.scalePoint()
          .domain(stages)
          .range([0, innerWidth])
          .padding(0.5);
        
        // Calculate clusters per stage and their counts
        const stageData = {};
        stages.forEach(stage => {
          stageData[stage] = {};
          clusters.forEach(cluster => {
            stageData[stage][cluster] = 0;
          });
          
          if (Array.isArray(this.data[stage])) {
            this.data[stage].forEach(cluster => {
              stageData[stage][cluster]++;
            });
          }
        });
        
        // Calculate flows between stages
        const flows = [];
        for (let i = 0; i < stages.length - 1; i++) {
          const sourceStage = stages[i];
          const targetStage = stages[i + 1];
          
          const sourceData = this.data[sourceStage];
          const targetData = this.data[targetStage];
          
          if (Array.isArray(sourceData) && Array.isArray(targetData)) {
            // Count how many items go from each source cluster to each target cluster
            const flowMap = {};
            
            // Only process up to the length of the shorter array
            const minLength = Math.min(sourceData.length, targetData.length);
            
            for (let j = 0; j < minLength; j++) {
              const sourceCluster = sourceData[j];
              const targetCluster = targetData[j];
              const flowKey = `${sourceCluster}-${targetCluster}`;
              
              if (!flowMap[flowKey]) {
                flowMap[flowKey] = {
                  source: sourceCluster,
                  target: targetCluster,
                  count: 0
                };
              }
              
              flowMap[flowKey].count++;
            }
            
            // Add flow data with stage information
            Object.values(flowMap).forEach(flow => {
              flows.push({
                sourceStage,
                targetStage,
                sourceCluster: flow.source,
                targetCluster: flow.target,
                count: flow.count
              });
            });
          }
        }
        
        // Calculate incoming and outgoing connections for each cluster in each stage
        const clusterConnections = {};
        stages.forEach(stage => {
          clusterConnections[stage] = {};
          clusters.forEach(cluster => {
            clusterConnections[stage][cluster] = {
              incoming: 0,
              outgoing: 0,
              total: 0
            };
          });
        });
        
        // Calculate outgoing connections
        flows.forEach(flow => {
          const { sourceStage, sourceCluster, count } = flow;
          clusterConnections[sourceStage][sourceCluster].outgoing += count;
        });
        
        // Calculate incoming connections
        flows.forEach(flow => {
          const { targetStage, targetCluster, count } = flow;
          clusterConnections[targetStage][targetCluster].incoming += count;
        });
        
        // Calculate total connections (incoming + outgoing)
        stages.forEach(stage => {
          clusters.forEach(cluster => {
            const connections = clusterConnections[stage][cluster];
            connections.total = connections.incoming + connections.outgoing;
          });
        });
        
        // Find the maximum total connection count to normalize node heights
        const maxConnectionCount = Math.max(
          1,
          d3.max(
            stages.map(stage => 
              d3.max(
                Object.values(clusterConnections[stage]).map(conn => conn.total)
              )
            )
          )
        );
        
        // Find the maximum number of clusters that need to be displayed in any stage
        const maxClustersInStage = Math.max(...stages.map(stage => 
          Object.values(stageData[stage]).filter(count => count > 0).length
        ));
        
        // Calculate the available height per cluster with spacing
        const spacing = 15; // Space between clusters
        const minNodeHeight = 20; // Minimum height for a node
        const maxNodeHeight = 80; // Maximum height for a node
        const availableHeight = Math.min(
          innerHeight / maxClustersInStage - spacing,
          maxNodeHeight
        );
        
        // Calculate vertical positions of clusters within each stage
        const stageClusterPositions = {};
        stages.forEach(stage => {
          const clusterPositions = {};
          let currentY = 0;
          
          // Sort clusters by count (descending)
          const sortedClusters = Object.entries(stageData[stage])
            .filter(([, count]) => count > 0)
            .sort((a, b) => {
              // First sort by connections.total (descending)
              const connA = clusterConnections[stage][a[0]].total;
              const connB = clusterConnections[stage][b[0]].total;
              return connB - connA || b[1] - a[1]; // Then by count as tiebreaker
            })
            .map(([cluster]) => cluster);
          
          // Assign positions
          sortedClusters.forEach(cluster => {
            // Use connection count for sizing instead of just item count
            const connectionTotal = clusterConnections[stage][cluster].total;
            const itemCount = stageData[stage][cluster];
            
            // Base height on connection count but factor in item count
            const weight = 0.7; // Weight for connection count (vs. item count)
            const normalizedConnections = connectionTotal / maxConnectionCount;
            const normalizedItems = itemCount / Math.max(1, d3.max(Object.values(stageData[stage])));
            
            // Combined score for sizing
            const sizeScore = weight * normalizedConnections + (1 - weight) * normalizedItems;
            
            // Scale height based on combined score
            const height = Math.max(
              minNodeHeight, 
              Math.min(
                maxNodeHeight,
                minNodeHeight + sizeScore * (availableHeight - minNodeHeight)
              )
            );
            
            clusterPositions[cluster] = {
              y: currentY,
              height,
              connections: clusterConnections[stage][cluster]
            };
            
            currentY += height + spacing;
          });
          
          stageClusterPositions[stage] = clusterPositions;
        });
        
        // Center the clusters vertically in each stage
        stages.forEach(stage => {
          const positions = stageClusterPositions[stage];
          const clusters = Object.keys(positions);
          if (clusters.length === 0) return;
          
          const totalHeight = clusters.reduce((sum, cluster) => 
            sum + positions[cluster].height, 0) + (clusters.length - 1) * spacing;
          
          const offset = (innerHeight - totalHeight) / 2;
          if (offset > 0) {
            clusters.forEach(cluster => {
              positions[cluster].y += offset;
            });
          }
        });
        
        // Find maximum flow count for scaling line widths
        const maxFlowCount = Math.max(1, d3.max(flows, flow => flow.count));
        
        // Create a scale for line widths
        const lineWidthScale = d3.scaleLinear()
          .domain([1, maxFlowCount])
          .range([1, 6])
          .clamp(true);
        
        // Keep track of active selection state
        let activeSelection = {
          stage: null,
          cluster: null,
          active: false
        };
        
        // Helper for clearing active selection
        const clearActiveSelection = () => {
          // Reset all nodes to normal
          chart.selectAll('.node')
            .attr('stroke', '#fff')
            .attr('stroke-width', 1);
          
          // Reset all flows to normal
          chart.selectAll('.flow')
            .style('opacity', 0.4)
            .style('stroke-width', d => lineWidthScale(d.count));
          
          activeSelection.active = false;
          
          // Clear info box
          this.clearInfoBox();
        };
        
        // Function to get incoming flows and source nodes for a cluster
        const getIncomingFlows = (targetStage, targetCluster) => {
          return flows.filter(flow => 
            flow.targetStage === targetStage && flow.targetCluster === targetCluster
          );
        };
        
        // Function to highlight incoming connections and source nodes
        const highlightIncoming = (targetStage, targetCluster, permanent = false) => {
          // Dim all flows
          chart.selectAll('.flow')
            .style('opacity', 0.1)
            .style('stroke-width', d => lineWidthScale(d.count));
          
          // Get all incoming flows
          const incomingFlows = getIncomingFlows(targetStage, targetCluster);
          
          // Highlight incoming flows
          incomingFlows.forEach(flow => {
            // Escape selectors to handle spaces correctly
            const sourceClass = this.escapeSelector(`flow-source-${flow.sourceCluster}`);
            const targetClass = this.escapeSelector(`flow-target-${flow.targetCluster}`);
            
            chart.selectAll(`.${sourceClass}.${targetClass}`)
              .filter(d => d.sourceStage === flow.sourceStage && d.targetStage === flow.targetStage)
              .style('opacity', 0.8)
              .style('stroke-width', d => lineWidthScale(d.count) + 2);
            
            // Highlight source nodes if permanent selection (clicked)
            if (permanent) {
              // Escape selector
              const nodeClass = this.escapeSelector(`node-${flow.sourceStage}-${flow.sourceCluster}`);
              chart.selectAll(`.${nodeClass}`)
                .attr('stroke', '#ff8800')
                .attr('stroke-width', 2);
            }
          });
          
          // Highlight the target node (escape selector)
          const nodeClass = this.escapeSelector(`node-${targetStage}-${targetCluster}`);
          chart.selectAll(`.${nodeClass}`)
            .attr('stroke', permanent ? '#ff0000' : '#ffcc00')
            .attr('stroke-width', 2);
        };
        
        // Draw stage labels
        chart.selectAll('.stage-label')
          .data(stages)
          .enter()
          .append('text')
          .attr('class', 'stage-label')
          .attr('x', d => xScale(d))
          .attr('y', -10)
          .attr('text-anchor', 'middle')
          .attr('font-weight', 'bold')
          .text(d => d);
        
        // Draw clusters as rectangles in each stage
        stages.forEach(stage => {
          const stageX = xScale(stage);
          const nodeWidth = 20;
          
          // Draw only clusters that have data
          Object.entries(stageClusterPositions[stage]).forEach(([cluster, position]) => {
            // Skip if height is too small
            if (position.height < 5) return;
            
            // Create a safe class name with no spaces or special characters
            const safeStage = stage.replace(/[^a-zA-Z0-9-]/g, '_');
            const safeCluster = cluster.replace(/[^a-zA-Z0-9-]/g, '_');
            const nodeClassName = `node node-${safeStage}-${safeCluster}`;
            
            // Store the original stage and cluster for data use
            const originalData = {
              stage,
              cluster
            };
            
            // Draw rectangle
            chart.append('rect')
              .attr('class', nodeClassName)
              .attr('x', stageX - nodeWidth/2)
              .attr('y', position.y)
              .attr('width', nodeWidth)
              .attr('height', position.height)
              .attr('fill', colorScale(cluster))
              .attr('stroke', '#fff')
              .attr('stroke-width', 1)
              .attr('rx', 3)
              .style('cursor', 'pointer')
              .datum(originalData) // Store the original data
              .on('mouseover', (event, d) => {
                // Only handle mouseover if not in active selection mode
                if (!activeSelection.active) {
                  // Update info box
                  const count = stageData[d.stage][d.cluster];
                  const connections = position.connections;
                  this.updateInfoBox(d.stage, d.cluster, count, connections);
                  
                  // Highlight incoming connections
                  highlightIncoming(d.stage, d.cluster);
                }
              })
              .on('mouseout', () => {
                // Only handle mouseout if not in active selection mode
                if (!activeSelection.active) {
                  // Clear info box
                  this.clearInfoBox();
                  
                  // Reset flows
                  chart.selectAll('.flow')
                    .style('opacity', 0.4)
                    .style('stroke-width', d => lineWidthScale(d.count));
                  
                  // Reset node highlighting
                  chart.selectAll('.node')
                    .attr('stroke', '#fff')
                    .attr('stroke-width', 1);
                }
              })
              .on('click', (event, d) => {
                // If we click on the already selected node, clear selection
                if (activeSelection.active && 
                    activeSelection.stage === d.stage && 
                    activeSelection.cluster === d.cluster) {
                  clearActiveSelection();
                  return;
                }
                
                // Clear any existing selection
                clearActiveSelection();
                
                // Set new active selection
                activeSelection = {
                  stage: d.stage,
                  cluster: d.cluster,
                  active: true
                };
                
                // Update info box with selected state
                const count = stageData[d.stage][d.cluster];
                const connections = position.connections;
                this.updateInfoBox(d.stage, d.cluster, count, connections, true);
                
                // Highlight with permanent selection
                highlightIncoming(d.stage, d.cluster, true);
              });
            
            // Add label
            chart.append('text')
              .attr('x', stageX)
              .attr('y', position.y + position.height/2 + 5)
              .attr('text-anchor', 'middle')
              .attr('font-size', '12px')
              .attr('fill', '#fff')
              .attr('pointer-events', 'none') // Make label transparent to mouse events
              .text(cluster);
          });
        });
        
        // Background click to clear selection
        svg.on('click', (event) => {
          // Ignore if the click was on a node
          if (event.target.classList.contains('node')) {
            return;
          }
          
          // Clear active selection
          if (activeSelection.active) {
            clearActiveSelection();
          }
        });
        
        // Draw flows between stages
        const linkGenerator = d3.linkHorizontal()
          .source(d => {
            const sourceX = xScale(d.sourceStage) + 10;
            const sourcePos = stageClusterPositions[d.sourceStage][d.sourceCluster];
            return [sourceX, sourcePos.y + sourcePos.height/2];
          })
          .target(d => {
            const targetX = xScale(d.targetStage) - 10;
            const targetPos = stageClusterPositions[d.targetStage][d.targetCluster];
            return [targetX, targetPos.y + targetPos.height/2];
          });
        
        chart.selectAll('.flow')
          .data(flows)
          .enter()
          .append('path')
          .attr('d', linkGenerator)
          .attr('fill', 'none')
          .attr('stroke', d => colorScale(d.sourceCluster))
          .attr('stroke-width', d => lineWidthScale(d.count))
          .attr('stroke-opacity', 0.4)
          .each(function(d) {
            // Create safe class names with no spaces
            const safeSourceCluster = d.sourceCluster.replace(/[^a-zA-Z0-9-]/g, '_');
            const safeTargetCluster = d.targetCluster.replace(/[^a-zA-Z0-9-]/g, '_');
            d3.select(this).attr('class', `flow flow-source-${safeSourceCluster} flow-target-${safeTargetCluster}`);
          })
          .on('mouseover', (event, d) => {
            // Only handle if not in active selection mode
            if (!activeSelection.active) {
              // Update info box with flow information
              this.selectedInfo = {
                title: `Flow: ${d.sourceStage} (${d.sourceCluster}) → ${d.targetStage} (${d.targetCluster})`,
                count: d.count,
                incoming: 'N/A',
                outgoing: 'N/A',
                total: d.count
              };
              
              // Highlight this flow
              d3.select(event.target)
                .style('opacity', 0.8)
                .style('stroke-width', lineWidthScale(d.count) + 2);
              
              // Highlight connected nodes (using safe class names)
              const safeSourceStage = d.sourceStage.replace(/[^a-zA-Z0-9-]/g, '_');
              const safeSourceCluster = d.sourceCluster.replace(/[^a-zA-Z0-9-]/g, '_');
              const safeTargetStage = d.targetStage.replace(/[^a-zA-Z0-9-]/g, '_');
              const safeTargetCluster = d.targetCluster.replace(/[^a-zA-Z0-9-]/g, '_');
              
              chart.select(`.node-${safeSourceStage}-${safeSourceCluster}`)
                .attr('stroke', '#ffcc00')
                .attr('stroke-width', 2);
              
              chart.select(`.node-${safeTargetStage}-${safeTargetCluster}`)
                .attr('stroke', '#ffcc00')
                .attr('stroke-width', 2);
            }
          })
          .on('mouseout', () => {
            // Only handle if not in active selection mode
            if (!activeSelection.active) {
              // Clear info box
              this.clearInfoBox();
              
              // Reset this flow
              chart.selectAll('.flow')
                .style('opacity', 0.4)
                .style('stroke-width', d => lineWidthScale(d.count));
              
              // Reset node highlighting
              chart.selectAll('.node')
                .attr('stroke', '#fff')
                .attr('stroke-width', 1);
            }
          });
        
        // // Add legend
        // const legend = svg.append('g')
        //   .attr('transform', `translate(${width - margin.right}, ${margin.top})`);
        
        // clusters.forEach((cluster, i) => {
        //   const legendItem = legend.append('g')
        //     .attr('transform', `translate(0, ${i * 20})`);
          
        //   legendItem.append('rect')
        //     .attr('width', 15)
        //     .attr('height', 15)
        //     .attr('fill', colorScale(cluster));
          
        //   legendItem.append('text')
        //     .attr('x', 20)
        //     .attr('y', 12)
        //     .text(`Cluster ${cluster}`);
        // });
        
        // Add instructions text
        svg.append('text')
          .attr('x', margin.left)
          .attr('y', height - 10)
          .attr('font-size', '12px')
          .attr('fill', '#666')
          .text('Hover: See details | Click: Select and highlight source nodes | Click background: Deselect');
      }
    }
  }
  </script>
  <style scoped>
  .chart-container {
    width: 100%;
    min-height: 500px;
    margin-bottom: 20px;
  }
  
  .info-box {
    width: 100%;
    min-height: 80px;
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 10px;
    background-color: #f9f9f9;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }
  
  .info-box h3 {
    margin-top: 0;
    margin-bottom: 10px;
    font-size: 16px;
    color: #333;
  }
  
  .info-stats {
    display: flex;
    justify-content: space-around;
  }
  
  .stat-item {
    text-align: center;
    padding: 0 10px;
  }
  
  .stat-value {
    font-size: 18px;
    font-weight: bold;
    color: #2c3e50;
  }
  
  .stat-label {
    font-size: 12px;
    color: #666;
  }
  
  .empty-info {
    color: #999;
    text-align: center;
    font-style: italic;
    padding: 20px;
  }
  </style>