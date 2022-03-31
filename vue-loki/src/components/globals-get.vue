<script>
export default {
  // data() {
  //   return {
  //     todoData: null
  //   }
  // },
  emits: ['response'],

  methods: {
    async fetchData() {
      this.todoData = null
      const requestOptions = {
        method: "POST",
      };
      const res = await fetch(`http://localhost:8000/fr/debug/inspect_globals`,requestOptions)
      const data = await res.json()
      // this.todoData = data
      console.log(`Fecthed news data successfully`)
      this.$emit('response', data)
    },
    
    test_alert() {
      alert('Test alert')
    }
  },

  mounted() {
    console.log(`Global variable component initiated`)
  },
}
</script>

<template>
  <a href="#" @click.prevent="fetchData">Show global variables</a>
  <!-- <p v-if="!todoData">Click to see the updated global variables...</p>
  <pre v-else>{{ todoData }}</pre> -->

  
</template>