<template>
  <router-view style="height:100%" />
</template>

<script>
import 'primevue/resources/themes/lara-light-green/theme.css'
import 'primeicons/primeicons.css'
import winsize from "@/tools/winsize";


export default {
  mixins: [winsize]
}

</script>

<style>
html, body, #app {
  height: 100%
}

#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}

nav {
  padding: 30px;
}

nav a {
  font-weight: bold;
  color: #2c3e50;
}

nav a.router-link-exact-active {
  color: #42b983;
}
</style>
