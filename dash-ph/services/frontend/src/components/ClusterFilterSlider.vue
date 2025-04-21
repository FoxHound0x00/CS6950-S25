<!-- ClusterFilterSlider.vue -->
<template>
    <div class="cluster-filter">
      <h3>Filter Clusters by Size:</h3>
      <div class="slider-container">
        <input 
          type="range" 
          class="size-slider" 
          min="1" 
          :max="maxClusterSize" 
          v-model.number="minClusterSize"
          @input="updateFilteredData" 
        />
        <div class="slider-value">{{ minClusterSize }} items</div>
      </div>
      <div class="filter-info">
        <p>Only showing clusters with {{ minClusterSize }} or more items</p>
        <div class="stats">
          <div class="stat-item">
            <span class="stat-label">Original data points:</span>
            <span class="stat-value">{{ totalOriginalItems }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Filtered data points:</span>
            <span class="stat-value">{{ totalFilteredItems }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Removed:</span>
            <span class="stat-value">{{ totalOriginalItems - totalFilteredItems }}</span>
          </div>
        </div>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    name: 'ClusterFilterSlider',
    props: {
      clusterData: {
        type: Object,
        required: true
      }
    },
    data() {
      return {
        minClusterSize: 1,
        maxClusterSize: 1,
        filteredData: {},
        totalOriginalItems: 0,
        totalFilteredItems: 0
      };
    },
    watch: {
      clusterData: {
        immediate: true,
        handler() {
          this.calculateMaxClusterSize();
          this.updateFilteredData();
        }
      }
    },
    methods: {
      calculateMaxClusterSize() {
        let max = 1;
        
        if (Object.keys(this.clusterData).length === 0) {
          return;
        }
        
        // For each stage, find the maximum count for any cluster
        Object.entries(this.clusterData).forEach(([stageName, clusterIds]) => {
          console.log("Not removing this since it'll trigger an error -- Stage Name: " + stageName)
          const counts = this.getCounts(clusterIds);
          const stageMax = Math.max(...Object.values(counts));
          if (stageMax > max) {
            max = stageMax;
          }
        });
        
        this.maxClusterSize = max;
        // Reset the min cluster size if needed
        if (this.minClusterSize > this.maxClusterSize) {
          this.minClusterSize = 1;
        }
      },
      
      // Helper to count occurrences of each element
      getCounts(array) {
        const counts = {};
        array.forEach(item => {
          counts[item] = (counts[item] || 0) + 1;
        });
        return counts;
      },
      
      updateFilteredData() {
        if (Object.keys(this.clusterData).length === 0) {
          this.filteredData = {};
          this.totalOriginalItems = 0;
          this.totalFilteredItems = 0;
          return;
        }
        
        // 1. For each stage, identify small clusters (counts < threshold)
        const smallClusters = new Map();
        
        Object.entries(this.clusterData).forEach(([stageName, clusterIds]) => {
          const counts = this.getCounts(clusterIds);
          
          // For each unique cluster ID, check if it's below the threshold
          Object.entries(counts).forEach(([clusterId, count]) => {
            if (count < this.minClusterSize) {
              if (!smallClusters.has(stageName)) {
                smallClusters.set(stageName, new Set());
              }
              smallClusters.get(stageName).add(clusterId);
            }
          });
        });
        
        // 2. Find all indexes to remove from each stage
        const indexesToRemove = new Set();
        
        Object.entries(this.clusterData).forEach(([stageName, clusterIds]) => {
          if (smallClusters.has(stageName)) {
            const smallClusterIds = smallClusters.get(stageName);
            
            clusterIds.forEach((id, index) => {
              if (smallClusterIds.has(id)) {
                indexesToRemove.add(index);
              }
            });
          }
        });
        
        // 3. Create filtered data by removing identified indexes from all stages
        const filtered = {};
        
        // Count the original items from the first stage (all stages should have same length)
        if (Object.keys(this.clusterData).length > 0) {
          const firstStageName = Object.keys(this.clusterData)[0];
          this.totalOriginalItems = this.clusterData[firstStageName].length;
        } else {
          this.totalOriginalItems = 0;
        }
        
        // Remove the indexes from all stages
        Object.entries(this.clusterData).forEach(([stageName, clusterIds]) => {
          filtered[stageName] = clusterIds.filter((_, index) => !indexesToRemove.has(index));
        });
        
        this.filteredData = filtered;
        
        // Count filtered items (should be the same for all stages)
        if (Object.keys(filtered).length > 0) {
          const firstStageName = Object.keys(filtered)[0];
          this.totalFilteredItems = filtered[firstStageName].length;
        } else {
          this.totalFilteredItems = 0;
        }
        
        // Emit the filtered data to the parent component
        this.$emit('update:filtered-data', filtered);
      }
    }
  }
  </script>
  
  <style scoped>
  .cluster-filter {
    margin-bottom: 30px;
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 6px;
    border: 1px solid #e9ecef;
  }
  
  .cluster-filter h3 {
    margin-top: 0;
    margin-bottom: 15px;
    font-size: 16px;
    color: #2c3e50;
  }
  
  .slider-container {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
  }
  
  .size-slider {
    flex: 1;
    height: 5px;
    background: linear-gradient(to right, #e1e1e1, #4caf50);
    border-radius: 5px;
    -webkit-appearance: none;
    appearance: none;
    outline: none;
    margin-right: 15px;
  }
  
  .size-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: #4caf50;
    cursor: pointer;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    transition: all 0.2s ease;
  }
  
  .size-slider::-webkit-slider-thumb:hover {
    background: #3d8b40;
    box-shadow: 0 3px 7px rgba(0, 0, 0, 0.3);
  }
  
  .size-slider::-moz-range-thumb {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: #4caf50;
    cursor: pointer;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    transition: all 0.2s ease;
    border: none;
  }
  
  .size-slider::-moz-range-thumb:hover {
    background: #3d8b40;
    box-shadow: 0 3px 7px rgba(0, 0, 0, 0.3);
  }
  
  .slider-value {
    width: 80px;
    text-align: right;
    font-weight: bold;
    color: #4caf50;
  }
  
  .filter-info {
    margin-top: 10px;
    font-size: 14px;
    color: #666;
  }
  
  .stats {
    display: flex;
    gap: 20px;
    margin-top: 10px;
  }
  
  .stat-item {
    display: flex;
    align-items: center;
  }
  
  .stat-label {
    margin-right: 5px;
  }
  
  .stat-value {
    font-weight: bold;
    color: #2c3e50;
  }
  
  @media (max-width: 768px) {
    .slider-container {
      flex-direction: column;
      align-items: stretch;
    }
    
    .slider-value {
      width: auto;
      text-align: left;
      margin-top: 5px;
    }
    
    .stats {
      flex-direction: column;
      gap: 5px;
    }
  }
  </style>