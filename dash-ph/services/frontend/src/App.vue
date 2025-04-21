<template>
  <div class="app">
    <h1>Cluster Flow Visualization</h1>
    
    <div class="dropdown-container">
      <div class="dropdown-item" v-for="(options, title) in dropdownOptions" :key="title">
        <label>{{ title }}</label>
        <select v-model="selectedOptions[title]" @change="updateClusterData">
          <option v-for="option in options" :key="option" :value="option">{{ option }}</option>
        </select>
      </div>
    </div>
    
    <ClusterSelectorSlider 
    :deathData="deathData"
    :availableStages="availableStages"
    @update:selected-stages="updateSelectedStages"
    @update:selected-clusters="updateSelectedClusters"
  />
    
    <ClusterFilterSlider 
      :cluster-data="filteredClusterData" 
      @update:filtered-data="updateClusterSizeFiltered"
    />
    
    <ClusterFlowChart :data="sizeFilteredClusterData" />

    <ForceDirectedGraph :data="sizeFilteredClusterData" />
  </div>
</template>

<script>
import ClusterFlowChart from './components/ClusterFlowChart.vue';
import ClusterFilterSlider from './components/ClusterFilterSlider.vue';
import ClusterSelectorSlider from './components/ClusterSelectorSlider.vue';
import ForceDirectedGraph from './components/ForceDirectedGraph.vue';
import axios from 'axios';
import * as d3 from 'd3';

export default {
  name: 'App',
  components: {
    ClusterFlowChart,
    ClusterFilterSlider,
    ClusterSelectorSlider,
    ForceDirectedGraph
  },
  data() {
    return {
      dropdownOptions: {},
      selectedOptions: {},
      rawClusterData: {},
      formattedClusterData: {},
      filteredClusterData: {},
      sizeFilteredClusterData: {},
      availableStages: [],
      deathData: {},
      selectedStages: [],
      brushChart: null
    };
  },
  created() {
    this.fetchConfig();
  },
  mounted() {
    // Initialize brush container after DOM is ready
    this.$nextTick(() => {
      this.initializeBrushChart();
    });
  },
  methods: {
    async fetchConfig() {
      try {
        const response = await axios.get('http://127.0.0.1:5000/config');
        this.dropdownOptions = response.data;
        
        // Initialize selected options
        this.selectedOptions = {};
        for (const [key, values] of Object.entries(this.dropdownOptions)) {
          if (values.length > 0) {
            this.selectedOptions[key] = values[0];
          }
        }
        
        this.updateClusterData();
      } catch (error) {
        console.error('Error fetching configuration:', error);
      }
    },
    async updateClusterData() {
      if (!this.selectedOptions['Distance Metric']) return;
      
      try {
        // Get data for the selected distance metric
        const response = await axios.get(`http://127.0.0.1:5000/data/${this.selectedOptions['Distance Metric']}`);
        
        // Store the raw data
        this.rawClusterData = response.data;
        
        // Format the data
        this.formatClusterData();
        
        // Update available stages
        this.availableStages = Object.keys(this.formattedClusterData);
        const deaths_ = await axios.get(`http://127.0.0.1:5000/cluster_data/${this.selectedOptions['Distance Metric']}`);
        this.deathData = deaths_.data;

        // Initially select all stages
        this.selectedStages = [...this.availableStages];
        
        // Update filtered data
        this.updateFilteredData();
        
        // Update brush chart with new data
        this.updateBrushChart();
      } catch (error) {
        console.error('Error fetching cluster data:', error);
      }
    },
    formatClusterData() {
      const formattedData = {};
      
      // Process each stage (key) in the raw data
      Object.entries(this.rawClusterData).forEach(([stage, clusterIds]) => {
        // Format stage name if it's a number (but not "Original Labels")
        let formattedStage = stage;
        if (stage !== "Original Labels" && !isNaN(parseFloat(stage))) {
          formattedStage = parseFloat(stage).toFixed(3);
        }
        
        // Simply convert numbers to strings
        formattedData[formattedStage] = clusterIds.map(id => String(id));
      });
      
      this.formattedClusterData = formattedData;
    },
    updateSelectedStages(stages) {
    this.selectedStages = stages;
    this.updateFilteredData();
  },
    updateFilteredData() {
      // Filter data to only include selected stages
      const filteredData = {};
      this.selectedStages.forEach(stage => {
        if (this.formattedClusterData[stage]) {
          filteredData[stage] = this.formattedClusterData[stage];
        }
      });
      
      this.filteredClusterData = filteredData;
      
      // Initialize the size-filtered data with the stage-filtered data
      this.sizeFilteredClusterData = { ...filteredData };
    },
    // Method to update selected clusters from ClusterSelectorSlider
    updateSelectedClusters(selectedClusters) {
      console.log("Selected clusters updated:", selectedClusters);
      // Implement handling of selected clusters if needed
    },
    // Method to update data after filtering by cluster size
    updateClusterSizeFiltered(filteredData) {
      this.sizeFilteredClusterData = filteredData;
    },
    // Initialize the brush chart
    initializeBrushChart() {
      // Make sure the container exists
      if (!this.$refs.brushContainer) return;
      
      // Create the basic structure for the brush chart
      this.brushChart = {};
      this.updateBrushChart();
    },
    // Update the brush chart with current available stages
    updateBrushChart() {
      if (!this.$refs.brushContainer || this.availableStages.length === 0) return;
      
      // Clear previous chart
      d3.select(this.$refs.brushContainer).selectAll("*").remove();
      
      // Chart dimensions
      const margin = { top: 20, right: 30, bottom: 50, left: 50 };
      const width = this.$refs.brushContainer.clientWidth || 800;
      const height = 150;
      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;
      
      // Create SVG
      const svg = d3.select(this.$refs.brushContainer)
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);
      
      // Prepare data
      const data = this.availableStages.map((stage, index) => ({
        stage, 
        index
      }));
      
      // Scales
      const x = d3.scaleLinear()
        .domain([0, this.availableStages.length - 1])
        .range([0, innerWidth]);
        // .range([innerWidth, 0]);

      
      // Create bars for each stage
      svg.selectAll('.stage-bar')
        .data(data)
        .enter()
        .append('rect')
        .attr('class', 'stage-bar')
        .attr('x', d => x(d.index))
        .attr('y', 0)
        .attr('width', innerWidth / this.availableStages.length)
        .attr('height', innerHeight)
        .attr('fill', '#69b3a2')
        .attr('opacity', 0.7);
      
      // X Axis - show indices
      svg.append('g')
        .attr('transform', `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x).ticks(Math.min(20, this.availableStages.length))
              .tickFormat(i => Math.floor(i)));
      
      // Brush
      const brush = d3.brushX()
        .extent([[0, 0], [innerWidth, innerHeight]])
        .on("end", brushed);
      
      svg.append("g")
        .attr("class", "brush")
        .call(brush);
      
      // Brush callback
      const self = this;
      // function brushed(event) {
      //   if (!event.selection) {
      //     // No selection, use all stages
      //     self.selectedStages = [...self.availableStages];
      //     self.updateFilteredData();
      //     return;
      //   }
        
      //   const [brushStart, brushEnd] = event.selection;
        
      //   // Convert brush positions to indices
      //   const startIndex = Math.floor(x.invert(brushStart));
      //   const endIndex = Math.ceil(x.invert(brushEnd));
        
      //   // Get stages in the selected range
      //   self.selectedStages = self.availableStages.filter((_, index) => 
      //     index >= startIndex && index <= endIndex
      //   );
        
      //   // Update data with selected stages
      //   self.updateFilteredData();
      // }
      function brushed(event) {
        if (!event.selection) {
          // No selection, use all stages
          self.selectedStages = [...self.availableStages];
          self.updateFilteredData();
          return;
        }
        
        const [brushStart, brushEnd] = event.selection;
        
        // Convert brush positions to indices
        const startIndex = Math.floor(x.invert(brushStart));
        const endIndex = Math.ceil(x.invert(brushEnd));
        
        // Get stages in the selected range (in reverse order)
        const brushedStages = self.availableStages
          .filter((_, index) => index >= startIndex && index <= endIndex)
          .reverse();
        
        // Always include the 0th index and attach reversed brushed stages
        const zeroStage = self.availableStages[0];
        self.selectedStages = zeroStage 
          ? [zeroStage, ...brushedStages.filter(stage => stage !== zeroStage)] 
          : [...brushedStages];
        
        // Update data with selected stages
        self.updateFilteredData();
      }
    }
  }
}
</script>

<style>
.app {
  font-family: Arial, sans-serif;
  max-width: 1000px;
  margin: 0 auto;
  padding: 20px;
}

h1 {
  text-align: center;
  margin-bottom: 30px;
  color: #2c3e50;
}

.dropdown-container {
  display: flex;
  justify-content: center;
  align-items: flex-end;
  flex-wrap: wrap;
  margin-bottom: 30px;
  gap: 20px;
}

.dropdown-item {
  flex: 1;
  min-width: 200px;
  max-width: 300px;
  margin-bottom: 15px;
}

.dropdown-item label {
  display: block;
  margin-bottom: 8px;
  font-weight: bold;
  color: #2c3e50;
  font-size: 14px;
}

.dropdown-item select {
  width: 100%;
  padding: 10px 15px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  font-size: 14px;
  background-color: white;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
  transition: all 0.3s ease;
  cursor: pointer;
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23555' d='M6 8.825L1.175 4 2.05 3.125 6 7.075 9.95 3.125 10.825 4z'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 10px center;
}

.dropdown-item select:hover {
  border-color: #c0c4cc;
}

.dropdown-item select:focus {
  outline: none;
  border-color: #409eff;
  box-shadow: 0 0 0 2px rgba(64, 158, 255, 0.2);
}

.stage-selector {
  margin-bottom: 30px;
  background-color: #f8f9fa;
  padding: 15px;
  border-radius: 6px;
  border: 1px solid #e9ecef;
}

.stage-selector h3 {
  margin-top: 0;
  margin-bottom: 15px;
  font-size: 16px;
  color: #2c3e50;
}

.brush-container {
  width: 100%;
  height: 150px;
}

.brush .overlay {
  cursor: crosshair;
}

/* For mobile devices */
@media (max-width: 768px) {
  .dropdown-container {
    flex-direction: column;
    align-items: stretch;
  }
  
  .dropdown-item {
    max-width: none;
  }
}
</style>