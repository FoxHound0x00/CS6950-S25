<template>
  <div class="home">
      <div class="dropdowns">
      <div v-for="(values, key) in config" :key="key">
        <label :for="key">{{ key }}</label>
        <select :id="key" v-model="selectedOptions[key]">
          <option v-for="option in values" :key="option" :value="option">
            {{ option }}
          </option>
        </select>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';


export default {
  components: {
  },
  data() {
    return {
      config: {}, 
      selectedOptions: {}
    };
  },
  methods: {
    async fetchConfig() {
      try {
        const response = await axios.get('/config'); // Fetch config
        console.log(response.data);
        this.config = response.data;

        // Initialize selectedOptions with null values for each dropdown
        this.selectedOptions = Object.keys(response.data).reduce((acc, key) => {
          acc[key] = null;
          return acc;
        }, {});
      } catch (error) {
        console.error('Error fetching config:', error);
      }
    }
  },
  mounted() {
    this.fetchConfig();
  }
};
</script>

<style scoped>
.dropdowns {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
}
</style>
