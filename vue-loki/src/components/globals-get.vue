<script>
export default {
  
  data() {
    return {
      data: 'no data'
    }
  },

  methods: {
    async fetchData() {
      const requestOptions = {
        method: "POST",
      };
      const res = await fetch(`http://localhost:8000/fr/debug/inspect_globals`,requestOptions)
      this.data = await res.json()
      console.log(`Fecthed news data successfully`)
      
    },
  },

  mounted() {
    console.log(`Global variable component initiated`)
    this.fetchData()
  },
}
</script>


<template>
{{ data }}
</template>