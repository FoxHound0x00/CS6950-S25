<template>
  <nav class="bg-gradient-to-r from-blue-500 to-green-400 shadow-xl rounded-b-2xl p-4 flex justify-between items-center border-b border-white/40">
    <div class="flex items-center space-x-4">
      <img src="/logo.png" alt="Logo" class="w-12 h-12 rounded-full shadow-md border-2 border-white/50" />
      <h1 class="text-white text-3xl font-bold drop-shadow-md">AeroNav</h1>
    </div>
    
    <ul class="flex w-full justify-center space-x-12 list-none">
      <li v-for="link in links" :key="link.name" class="flex">
        <router-link 
          :to="link.path" 
          class="px-8 py-3 rounded-xl text-white text-lg font-semibold bg-white/20 backdrop-blur-md transition-all duration-300 hover:bg-green-300 hover:text-blue-900 shadow-lg border border-white/30 no-underline">
          {{ link.name }}
        </router-link>
      </li>
    </ul>
  </nav>
</template>

<script>
export default {
  data() {
    return {
      links: [
        { name: "Home", path: "/" },
        { name: "About", path: "/about" },
        { name: "Services", path: "/services" },
        { name: "Contact", path: "/contact" }
      ]
    };
  }
};
</script>

<style scoped>
nav {
  background: linear-gradient(135deg, rgba(0, 200, 255, 0.9), rgba(0, 255, 150, 0.9));
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
  backdrop-filter: blur(15px);
  border-bottom-left-radius: 16px;
  border-bottom-right-radius: 16px;
  padding: 16px 40px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

ul {
  display: flex;
  align-items: center;
  justify-content: space-around;
  width: 100%;
  list-style: none;
}

li {
  display: flex;
  align-items: center;
}

router-link {
  text-decoration: none;
}
</style>
