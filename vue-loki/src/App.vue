<template>
  <div class="wrapper">
    <header class="header"><Home msg="Welcome to Loki MVP"/></header>
    <aside class="aside aside-1">
      <GlobalsGet @response="(msg) => MainContent = msg"/>
    </aside>
    <article class="main">
      <p v-if="!MainContent">Use the left menu</p>
      <pre v-else>{{ MainContent }}</pre>
    </article>
  </div>
  
</template>


<script>
// imports
import Home from './components/home.vue'
import GlobalsGet from './components/globals-get.vue'

// exports
export default {
  name: 'App',
  components: {
    Home,
    GlobalsGet
  },

  data(){
    return {
      MainContent: null
    }
  },

  created() {
    this.const_test1 = 'ok';
  },

  mounted() {
    console.log(`Main app initiated.`)
    console.log(`Constant "const_test1"= "${this.const_test1}"`)
  },
}
</script>


<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  /* text-align: center; */
  color: #2c3e50;
  /* margin-top: 20px; */
}

.wrapper {
  display: flex;  
  flex-flow: row wrap;
  /* text-align: center;  */
}

.wrapper > * {
  padding: 20px;
  flex: 1 100%;
}

.header {
  text-align: center;
}

.footer {
/*   background: lightgreen; */
}

.main {
  text-align: left;
/*   background: deepskyblue; */
}

.aside-1 {
  background: #eee;
}

@media all and (min-width: 600px) {
  .aside { flex: 1 0 0; }
}

@media all and (min-width: 800px) {
  .main    { flex: 3 0px; }
  .aside-1 { order: 1; } 
  .main    { order: 2; }
  .footer  { order: 4; }
}

body {
  padding: 1em; 
}
</style>
