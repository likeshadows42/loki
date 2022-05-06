<script>
import compGroupItemsCanvas from './img-groups-item-canvas.vue'
export default {
  components: {
      compGroupItemsCanvas
  },

  data() {
    return {
      imgs: null,
      all_grouped: false
    }
  },

  methods: {
    async fetchData() {
      const requestOptions = {
        method: "POST",
      };
      const res = await fetch(`http://127.0.0.1:8000/fr/facerep/get_ungrouped`,requestOptions)
      this.imgs = await res.json()
    },

    async fetchImg(item_id, person_id) {
        console.log(item_id+", "+person_id)
    },

    checkGroup(num) {
      if(num == -1) {
        return true
      } else {
        this.all_grouped = true
        return false
      }
    }
  },  

  mounted() {
    this.fetchData()
  },
}
</script>


<template>
<h2>List all untagged images</h2>

<span v-for="img in imgs" :key="img.id">
    <compGroupItemsCanvas :item="img" @parent-handler="fetchImg"></compGroupItemsCanvas>
</span>
<p v-if="imgs == true">No ungrouped images</p>

</template>


<style scoped>
.thumb {
    max-width: 80px;
}
</style>