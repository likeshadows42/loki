<script>
export default {
  data() {
    return {
      todoData: null
    }
  },
  methods: {
    async fetchData() {
      this.todoData = null
      const requestOptions = {
        method: "POST",
      };
      const res = await fetch(`http://localhost:8000/fr/debug/inspect_globals`,requestOptions)
      const data = await res.json()
      // console.log(this.todoData)
      this.todoData = data
    }
  },

  // mounted() {
  //   this.fetchData()
  // },
  // watch: {
  //   todoId() {
  //     this.fetchData()
  //   }
  // }
}
</script>

<template>
  <!-- <button @click="todoId++">Fetch next todo</button> -->
  <button @click="fetchData">Reload global variables</button>
  <p v-if="!todoData">Click to see the updated global variables...</p>
  <pre v-else>{{ todoData }}</pre>
</template>