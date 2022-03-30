<template>
  <!-- <img alt="Vue logo" src="./assets/logo.png"> -->
  <Home msg="Welcome to Loki MVP"/>
  <GlobalsGet/>
  <!-- <Test/> -->
</template>

<script>
import Home from './components/home.vue'
import GlobalsGet from './components/globals-get.vue'
// import Test from './components/test-one.vue'

export default {
  name: 'App',
  components: {
    Home,
    GlobalsGet
    // Test
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  /* text-align: center; */
  color: #2c3e50;
  margin-top: 20px;
}
</style>
